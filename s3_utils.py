import os
import logging
import traceback
import boto3
from botocore.exceptions import (
    ClientError,
    NoCredentialsError,
    PartialCredentialsError,
    EndpointResolutionError,
    ConnectTimeoutError,
    ReadTimeoutError,
)

logger = logging.getLogger(__name__)

s3_client = boto3.client("s3", verify=False)

# Max allowed /tmp file size: 400 MB (safe limit within Lambda's 512 MB /tmp)
MAX_FILE_SIZE_BYTES = 400 * 1024 * 1024

# Supported file extensions
SUPPORTED_EXTENSIONS = {"pdf", "docx", "doc"}


def download_document(bucket: str, key: str) -> str:
    """
    Download an S3 object to /tmp and return the local file path.

    Args:
        bucket: S3 bucket name.
        key:    S3 object key.

    Returns:
        Absolute path to the downloaded file in /tmp.

    Raises:
        ValueError:       Missing/invalid args, unsupported file type, or file too large.
        PermissionError:  No AWS credentials or access denied.
        FileNotFoundError: Object does not exist in S3.
        ConnectionError:  Network/timeout reaching S3.
        RuntimeError:     Any other S3 client error.
    """
    logger.info(f"[s3] download_document: bucket='{bucket}' key='{key}'")

    # --- Input validation ---
    try:
        if not bucket or not isinstance(bucket, str):
            raise ValueError(f"Invalid bucket name: {bucket!r}")
        if not key or not isinstance(key, str):
            raise ValueError(f"Invalid S3 key: {key!r}")

        file_name = key.split("/")[-1]
        if not file_name:
            raise ValueError(f"S3 key does not resolve to a file name: {key!r}")

        extension = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
        if extension not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '.{extension}' for key '{key}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
    except ValueError:
        raise
    except Exception as e:
        logger.error(
            f"[s3] download_document: input validation failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise ValueError(f"Input validation error: {e}") from e

    # --- Head object (check existence, size, credentials) ---
    try:
        head = s3_client.head_object(Bucket=bucket, Key=key)
        logger.info(
            f"[s3] download_document: head_object OK — "
            f"ContentLength={head.get('ContentLength', 'unknown')} "
            f"ContentType={head.get('ContentType', 'unknown')}"
        )
    except NoCredentialsError as e:
        logger.error(
            f"[s3] download_document: NO CREDENTIALS — "
            f"bucket='{bucket}' key='{key}'\n{traceback.format_exc()}"
        )
        raise PermissionError(
            "AWS credentials not found. Configure via environment variables, "
            "~/.aws/credentials, or an IAM role."
        ) from e
    except PartialCredentialsError as e:
        logger.error(
            f"[s3] download_document: PARTIAL CREDENTIALS — "
            f"error={e}\n{traceback.format_exc()}"
        )
        raise PermissionError(
            f"Incomplete AWS credentials: {e}. Ensure both access key and secret are set."
        ) from e
    except (ConnectTimeoutError, ReadTimeoutError) as e:
        logger.error(
            f"[s3] download_document: TIMEOUT on head_object — "
            f"bucket='{bucket}' key='{key}' error={e}\n{traceback.format_exc()}"
        )
        raise ConnectionError(f"Timed out connecting to S3 for s3://{bucket}/{key}: {e}") from e
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg  = e.response["Error"].get("Message", "")
        logger.error(
            f"[s3] download_document: ClientError on head_object — "
            f"code='{error_code}' msg='{error_msg}' "
            f"bucket='{bucket}' key='{key}'\n{traceback.format_exc()}"
        )
        if error_code == "404":
            raise FileNotFoundError(f"Object not found: s3://{bucket}/{key}") from e
        if error_code == "403":
            raise PermissionError(
                f"Access denied to s3://{bucket}/{key}. "
                "Check IAM permissions (s3:HeadObject, s3:GetObject)."
            ) from e
        if error_code == "NoSuchBucket":
            raise FileNotFoundError(f"Bucket does not exist: '{bucket}'") from e
        raise RuntimeError(
            f"S3 HeadObject failed for s3://{bucket}/{key} — "
            f"Code: {error_code}, Message: {error_msg}"
        ) from e
    except Exception as e:
        logger.error(
            f"[s3] download_document: unexpected error on head_object — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Unexpected error checking s3://{bucket}/{key}: {e}") from e

    # --- File size check ---
    try:
        file_size = head.get("ContentLength", 0)
        if file_size == 0:
            logger.warning(f"[s3] download_document: s3://{bucket}/{key} reports 0 bytes — file may be empty.")
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File too large: {file_size:,} bytes exceeds /tmp limit of "
                f"{MAX_FILE_SIZE_BYTES:,} bytes. Key: {key}"
            )
    except ValueError:
        raise
    except Exception as e:
        logger.error(
            f"[s3] download_document: file size check failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"File size validation error: {e}") from e

    # --- Download ---
    local_path = os.path.join("/tmp", file_name)
    logger.info(f"[s3] download_document: downloading s3://{bucket}/{key} → {local_path} ({file_size:,} bytes)")

    try:
        s3_client.download_file(bucket, key, local_path)
    except (ConnectTimeoutError, ReadTimeoutError) as e:
        logger.error(
            f"[s3] download_document: TIMEOUT during download — "
            f"bucket='{bucket}' key='{key}' error={e}\n{traceback.format_exc()}"
        )
        raise ConnectionError(
            f"Timed out downloading s3://{bucket}/{key}: {e}"
        ) from e
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        error_msg  = e.response["Error"].get("Message", "")
        logger.error(
            f"[s3] download_document: ClientError during download — "
            f"code='{error_code}' msg='{error_msg}'\n{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"S3 download failed for s3://{bucket}/{key} — "
            f"Code: {error_code}, Message: {error_msg}"
        ) from e
    except OSError as e:
        logger.error(
            f"[s3] download_document: OS ERROR writing to {local_path} — "
            f"error={e}\n{traceback.format_exc()}"
        )
        raise OSError(
            f"Cannot write to {local_path}. /tmp may be full or permission denied: {e}"
        ) from e
    except Exception as e:
        logger.error(
            f"[s3] download_document: unexpected error during download — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Unexpected download error for s3://{bucket}/{key}: {e}") from e

    # --- Verify download ---
    try:
        if not os.path.exists(local_path):
            raise RuntimeError(f"Download appeared to succeed but {local_path} does not exist.")
        if os.path.getsize(local_path) == 0:
            raise RuntimeError(f"Downloaded file is empty: {local_path}")
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(
            f"[s3] download_document: post-download verification failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Download verification failed: {e}") from e

    logger.info(f"[s3] download_document: SUCCESS — {local_path}")
    return local_path


def get_object_tags(bucket: str, key: str) -> dict:
    """
    Fetch S3 object tags and return them as a flat metadata dict.

    Expected tags:
        department     → e.g. "HR", "Finance"
        policy_type    → e.g. "leave", "handbook"
        effective_date → e.g. "2024-01-01" (ISO date string)

    Missing tags default to "unknown" / None. Tag fetch failures are
    non-fatal — defaults are used so ingestion can continue.

    Args:
        bucket: S3 bucket name.
        key:    S3 object key.

    Returns:
        dict with keys: department, policy_type, effective_date, file_type.

    Raises:
        ValueError: Missing/invalid args.
    """
    logger.info(f"[s3] get_object_tags: bucket='{bucket}' key='{key}'")

    if not bucket or not isinstance(bucket, str):
        raise ValueError(f"Invalid bucket name: {bucket!r}")
    if not key or not isinstance(key, str):
        raise ValueError(f"Invalid S3 key: {key!r}")

    raw_tags = {}
    try:
        response = s3_client.get_object_tagging(Bucket=bucket, Key=key)
        raw_tags = {tag["Key"]: tag["Value"] for tag in response.get("TagSet", [])}
        logger.info(f"[s3] get_object_tags: fetched {len(raw_tags)} tag(s) for s3://{bucket}/{key}")
    except NoCredentialsError as e:
        logger.warning(
            f"[s3] get_object_tags: NO CREDENTIALS for tag fetch — "
            f"using defaults. error={e}\n{traceback.format_exc()}"
        )
    except ClientError as e:
        error_code = e.response["Error"]["Code"]
        logger.warning(
            f"[s3] get_object_tags: ClientError code='{error_code}' "
            f"for s3://{bucket}/{key} — using defaults.\n{traceback.format_exc()}"
        )
    except Exception as e:
        logger.warning(
            f"[s3] get_object_tags: unexpected error — "
            f"type={type(e).__name__} error={e} — using defaults.\n{traceback.format_exc()}"
        )

    try:
        file_name  = key.split("/")[-1]
        file_type  = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "unknown"

        result = {
            "department":     raw_tags.get("department", "unknown"),
            "policy_type":    raw_tags.get("policy_type", "unknown"),
            "effective_date": raw_tags.get("effective_date", None),
            "file_type":      file_type,
        }
    except Exception as e:
        logger.error(
            f"[s3] get_object_tags: failed to build metadata dict — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to build metadata for key '{key}': {e}") from e

    logger.info(f"[s3] get_object_tags: result={result}")
    return result


def cleanup_tmp(local_path: str) -> None:
    """
    Remove the main document from /tmp and any page images generated
    during PDF processing (page_0.png, page_1.png, …).

    Args:
        local_path: Path to the primary file to delete.

    Raises:
        ValueError: local_path is None or empty.
    """
    logger.info(f"[s3] cleanup_tmp: cleaning up local_path='{local_path}'")

    if not local_path:
        raise ValueError("cleanup_tmp received an empty local_path — nothing to clean up.")

    try:
        _safe_remove(local_path)
    except Exception as e:
        logger.warning(
            f"[s3] cleanup_tmp: failed to remove main file '{local_path}' — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )

    # Remove any page images created by the PDF parser
    tmp_dir = "/tmp"
    try:
        entries = os.listdir(tmp_dir)
    except OSError as e:
        logger.warning(
            f"[s3] cleanup_tmp: could not list {tmp_dir} for page image cleanup — "
            f"error={e}\n{traceback.format_exc()}"
        )
        return

    for fname in entries:
        if fname.startswith("page_") and fname.endswith(".png"):
            try:
                _safe_remove(os.path.join(tmp_dir, fname))
            except Exception as e:
                logger.warning(
                    f"[s3] cleanup_tmp: failed to remove page image '{fname}' — "
                    f"type={type(e).__name__} error={e}"
                )

    logger.info(f"[s3] cleanup_tmp: cleanup complete for '{local_path}'")


def _safe_remove(path: str) -> None:
    """Remove a file, logging a warning on failure rather than raising."""
    try:
        os.remove(path)
        logger.debug(f"[s3] _safe_remove: removed {path}")
    except FileNotFoundError:
        logger.debug(f"[s3] _safe_remove: file already gone (skipping): {path}")
    except PermissionError as e:
        logger.warning(f"[s3] _safe_remove: permission denied removing {path}: {e}")
    except OSError as e:
        logger.warning(f"[s3] _safe_remove: could not remove {path}: {e}")
