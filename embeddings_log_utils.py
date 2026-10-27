"""
embeddings_log_utils.py
-----------------------
Append embedded chunks to a single persistent pickle file so every
document processed accumulates in one place.

Fixed file name:  embeddings_store.pkl  (in the project dir locally,
                  or /tmp when running inside AWS Lambda)

Stored object: list of dicts, growing with every run:
{
    "chunk_id":       "...",
    "document_id":    "...",
    "file_name":      "policy.pdf",
    "department":     "HR",
    "policy_type":    "leave",
    "effective_date": "2024-01-01",
    "file_type":      "pdf",
    "ingested_at":    "2024-10-22T10:30:00+00:00",
    "source_s3_key":  "referral.pdf",
    "chunk_index":    0,
    "text_content":   "...",
    "embedding_dim":  1536,
    "embedding":      [0.012, -0.034, ...]
}

To load in Python:
    import pickle
    with open("embeddings_store.pkl", "rb") as f:
        all_chunks = pickle.load(f)
"""

import json
import pickle
import logging
import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Single fixed file — grows across runs
STORE_FILENAME = "embeddings_store.pkl"
OUTPUT_DIR     = "/tmp" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else "."


def save_embeddings_log(
    chunks: List[Dict[str, Any]],
    document_id: str,
    output_dir: str = OUTPUT_DIR,
) -> str:
    """
    Append embedded chunks to the persistent store file.

    If the store file already exists its contents are loaded first,
    then the new chunks are added and the whole list is saved back.

    Args:
        chunks:      List of chunk dicts that already contain an "embedding" key.
        document_id: Unique ID for this ingestion run (used only for logging).
        output_dir:  Directory for the store file. Defaults to project dir locally,
                     /tmp inside Lambda.

    Returns:
        Absolute path of the store file.

    Raises:
        ValueError: chunks is None/empty or document_id is blank.
        OSError:    Cannot read/write the store file.
    """
    if not chunks or not isinstance(chunks, list):
        raise ValueError("save_embeddings_log: chunks must be a non-empty list.")
    if not document_id or not isinstance(document_id, str):
        raise ValueError(f"save_embeddings_log: invalid document_id: {document_id!r}")

    os.makedirs(output_dir, exist_ok=True)
    store_path = os.path.join(output_dir, STORE_FILENAME)

    # Load existing records if the store already exists
    existing: List[Dict[str, Any]] = []
    if os.path.exists(store_path):
        try:
            with open(store_path, "rb") as f:
                existing = pickle.load(f)
            if not isinstance(existing, list):
                logger.warning(f"Store file had unexpected type — starting fresh.")
                existing = []
        except Exception as e:
            logger.warning(f"Could not read existing store ({e}) — starting fresh.")
            existing = []

    new_records = [_to_log_record(chunk) for chunk in chunks]
    combined    = existing + new_records

    try:
        with open(store_path, "wb") as f:
            pickle.dump(combined, f)
    except OSError as e:
        raise OSError(f"Failed to write embeddings store to {store_path}: {e}") from e

    size_kb = os.path.getsize(store_path) / 1024
    logger.info(json.dumps({
        "stage":            "embeddings_log_saved",
        "document_id":      document_id,
        "store_path":       store_path,
        "new_chunks":       len(new_records),
        "total_chunks":     len(combined),
        "file_size_kb":     round(size_kb, 1),
    }))
    print(
        f"\n📄 Embeddings store updated → {store_path}  "
        f"(+{len(new_records)} new  |  {len(combined)} total  |  {size_kb:.1f} KB)\n"
    )
    return store_path


def _to_log_record(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a chunk dict into the canonical store record shape."""
    embedding = chunk.get("embedding", [])
    return {
        "chunk_id":       chunk.get("chunk_id"),
        "document_id":    chunk.get("document_id"),
        "file_name":      chunk.get("file_name"),
        "department":     chunk.get("department"),
        "policy_type":    chunk.get("policy_type"),
        "effective_date": chunk.get("effective_date"),
        "file_type":      chunk.get("file_type"),
        "ingested_at":    chunk.get("ingested_at"),
        "source_s3_key":  chunk.get("source_s3_key"),
        "chunk_index":    chunk.get("chunk_index"),
        "text_content":   chunk.get("text_content"),
        "embedding_dim":  len(embedding),
        "embedding":      embedding,
    }


def upload_store_to_s3(store_path: str, bucket: str, s3_prefix: str = "embeddings-store") -> str:
    """
    Upload the local embeddings store pkl to S3.

    Used as a fallback when OpenSearch is not configured — the pkl lands in S3
    so it is not lost when the Lambda container shuts down.

    Args:
        store_path: Local path to the pkl file.
        bucket:     Destination S3 bucket.
        s3_prefix:  Key prefix inside the bucket (default: "embeddings-store").

    Returns:
        The S3 key the file was uploaded to.

    Raises:
        PermissionError: No credentials or access denied.
        RuntimeError:    Any other S3 failure.
    """
    filename = os.path.basename(store_path)
    s3_key   = f"{s3_prefix}/{filename}"

    logger.info(f"Uploading embeddings store → s3://{bucket}/{s3_key}")
    try:
        s3 = boto3.client("s3", verify=False)
        s3.upload_file(store_path, bucket, s3_key)
    except NoCredentialsError as e:
        raise PermissionError(
            "No AWS credentials found while uploading embeddings store."
        ) from e
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "403":
            raise PermissionError(
                f"Access denied uploading to s3://{bucket}/{s3_key}. "
                "Check s3:PutObject permission."
            ) from e
        raise RuntimeError(
            f"S3 upload failed for s3://{bucket}/{s3_key} — Code: {code}"
        ) from e

    logger.info(json.dumps({
        "stage":    "embeddings_store_uploaded_to_s3",
        "s3_key":   s3_key,
        "bucket":   bucket,
    }))
    print(f"☁️  Embeddings store uploaded → s3://{bucket}/{s3_key}\n")
    return s3_key


def delete_embeddings_by_s3_key(
    s3_key: str,
    bucket: str,
    output_dir: str = OUTPUT_DIR,
    s3_prefix: str = "embeddings-store",
) -> int:
    """
    Remove all chunks for a given S3 object key from the persistent store.

    Strategy:
      1. Try to load the store from the local output_dir (warm Lambda container).
      2. If not found locally, download it from S3 first.
      3. Filter out every record whose source_s3_key matches s3_key.
      4. Write the filtered list back to disk and re-upload to S3.

    Args:
        s3_key:     The S3 object key of the deleted document (e.g. "hr/policy.pdf").
        bucket:     The S3 bucket — used to download/re-upload the store.
        output_dir: Local directory for the store file.
        s3_prefix:  S3 key prefix where the store is kept.

    Returns:
        Number of chunks removed.

    Raises:
        ValueError:      s3_key or bucket is blank.
        RuntimeError:    Store file not found locally or in S3.
        OSError:         Cannot read/write the local store file.
        PermissionError: No AWS credentials or access denied.
    """
    if not s3_key or not isinstance(s3_key, str):
        raise ValueError(f"delete_embeddings_by_s3_key: invalid s3_key: {s3_key!r}")
    if not bucket or not isinstance(bucket, str):
        raise ValueError(f"delete_embeddings_by_s3_key: invalid bucket: {bucket!r}")

    os.makedirs(output_dir, exist_ok=True)
    store_path = os.path.join(output_dir, STORE_FILENAME)

    # ── Step 1: Ensure we have a local copy ────────────────────────────────
    if not os.path.exists(store_path):
        remote_key = f"{s3_prefix}/{STORE_FILENAME}"
        logger.info(f"Store not found locally — downloading s3://{bucket}/{remote_key}")
        try:
            s3 = boto3.client("s3", verify=False)
            s3.download_file(bucket, remote_key, store_path)
        except NoCredentialsError as e:
            raise PermissionError(
                "No AWS credentials found while downloading embeddings store."
            ) from e
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey"):
                raise RuntimeError(
                    f"Embeddings store not found at s3://{bucket}/{remote_key}. "
                    "Nothing to delete."
                ) from e
            if code == "403":
                raise PermissionError(
                    f"Access denied downloading s3://{bucket}/{remote_key}."
                ) from e
            raise RuntimeError(
                f"S3 download failed for s3://{bucket}/{remote_key} — Code: {code}"
            ) from e

    # ── Step 2: Load ────────────────────────────────────────────────────────
    try:
        with open(store_path, "rb") as f:
            all_chunks: List[Dict[str, Any]] = pickle.load(f)
        if not isinstance(all_chunks, list):
            logger.warning("Store file had unexpected type — treating as empty.")
            all_chunks = []
    except Exception as e:
        raise OSError(f"Failed to read embeddings store from {store_path}: {e}") from e

    before = len(all_chunks)

    # ── Step 3: Filter ──────────────────────────────────────────────────────
    filtered = [c for c in all_chunks if c.get("source_s3_key") != s3_key]
    removed  = before - len(filtered)

    if removed == 0:
        logger.warning(
            json.dumps({
                "stage":   "delete_embeddings_no_match",
                "s3_key":  s3_key,
                "message": "No chunks matched — store unchanged.",
            })
        )
        return 0

    # ── Step 4: Save locally ────────────────────────────────────────────────
    try:
        with open(store_path, "wb") as f:
            pickle.dump(filtered, f)
    except OSError as e:
        raise OSError(f"Failed to write filtered store to {store_path}: {e}") from e

    size_kb = os.path.getsize(store_path) / 1024
    logger.info(json.dumps({
        "stage":          "delete_embeddings_saved",
        "s3_key":         s3_key,
        "chunks_removed": removed,
        "chunks_remaining": len(filtered),
        "file_size_kb":   round(size_kb, 1),
    }))

    # ── Step 5: Re-upload to S3 ─────────────────────────────────────────────
    try:
        upload_store_to_s3(store_path, bucket, s3_prefix)
    except Exception as e:
        # Non-fatal: local store is correct; log and continue.
        logger.warning(
            json.dumps({
                "stage":   "delete_embeddings_upload_failed",
                "s3_key":  s3_key,
                "error":   str(e),
                "hint":    "Local store updated but S3 re-upload failed.",
            })
        )

    print(
        f"\n🗑️  Embeddings deleted for '{s3_key}'  "
        f"(removed {removed}  |  {len(filtered)} remaining  |  {size_kb:.1f} KB)\n"
    )
    return removed
