
import json
import pickle
import logging
import os
import re
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

STORE_SUBDIR = "embeddings-store"          # folder name, used both locally and as the S3 prefix
OUTPUT_DIR   = "/tmp" if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else "."


def _sanitize_key(s3_key: str) -> str:
    """Turn an S3 key into a filesystem-safe fragment: 'hr/policy.pdf' -> 'hr__policy.pdf'."""
    safe = s3_key.strip().replace("/", "__")
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", safe)
    return safe or "unknown-key"


def _store_dir(output_dir: str) -> str:
    path = os.path.join(output_dir, STORE_SUBDIR)
    os.makedirs(path, exist_ok=True)
    return path


def save_embeddings_log(
    chunks: List[Dict[str, Any]],
    document_id: str,
    output_dir: str = OUTPUT_DIR,
    bucket: str = None,
    s3_prefix: str = STORE_SUBDIR,
    replace_existing: bool = True,
) -> str:
    """
    Write this document's embedded chunks to their OWN pickle file — never
    touches or depends on any other document's pkl, so there is no
    read-modify-write race between Lambda containers and nothing can be
    silently overwritten.

    File name: "{sanitized(source_s3_key)}__{document_id}.pkl"

    If a document with the same "source_s3_key" was already ingested before
    (e.g. the file was re-uploaded/overwritten in S3), its OLDER pkl(s) are
    deleted first — so the store ends up with exactly one pkl per
    currently-existing S3 object, not one per historical upload.

    Args:
        chunks:      List of chunk dicts that already contain an "embedding" key.
                     All chunks are expected to belong to the same document
                     (same "source_s3_key" / "document_id").
        document_id: Unique ID for this ingestion run — becomes part of the filename.
        output_dir:  Base directory for the local store. Defaults to project
                     dir locally, /tmp inside Lambda. The pkl is written to
                     "{output_dir}/{s3_prefix}/{filename}".
        bucket:      S3 bucket to upload the pkl to. If omitted, only the
                     local file is written (no S3 round-trip), and no old
                     remote pkl can be cleaned up either.
        s3_prefix:   S3 key prefix / local subfolder name (default: "embeddings-store").
        replace_existing: If True (default), delete any prior pkl(s) for this
                     document's "source_s3_key" before writing the new one.
                     Set False to keep full upload history instead.

    Returns:
        Absolute local path of the pkl file that was written.

    Raises:
        ValueError: chunks is None/empty or document_id is blank.
        OSError:    Cannot write the local pkl file.
    """
    if not chunks or not isinstance(chunks, list):
        raise ValueError("save_embeddings_log: chunks must be a non-empty list.")
    if not document_id or not isinstance(document_id, str):
        raise ValueError(f"save_embeddings_log: invalid document_id: {document_id!r}")

    source_s3_key = chunks[0].get("source_s3_key") or "unknown-key"
    file_name     = chunks[0].get("file_name") or source_s3_key

    filename   = f"{_sanitize_key(source_s3_key)}__{document_id}.pkl"
    store_dir  = _store_dir(output_dir)
    store_path = os.path.join(store_dir, filename)

    # ── Replace, don't accumulate: drop this document's previous pkl(s) ────
    # New document_id each run means the new filename never collides with an
    # old one, so without this step a re-uploaded file would pile up pkls
    # instead of being replaced.
    if replace_existing:
        try:
            files_removed, _ = _purge_pkls_for_key(
                source_s3_key, bucket=bucket, output_dir=output_dir, s3_prefix=s3_prefix,
            )
            if files_removed:
                logger.info(json.dumps({
                    "stage":          "embeddings_prior_pkl_replaced",
                    "document_id":    document_id,
                    "source_s3_key":  source_s3_key,
                    "files_replaced": files_removed,
                }))
        except Exception as e:
            logger.warning(f"Could not purge prior pkl(s) for '{source_s3_key}' ({e}) — continuing anyway.")

    records = [_to_log_record(chunk) for chunk in chunks]

    try:
        with open(store_path, "wb") as f:
            pickle.dump(records, f)
    except OSError as e:
        raise OSError(f"Failed to write embeddings pkl to {store_path}: {e}") from e

    size_kb = os.path.getsize(store_path) / 1024
    logger.info(json.dumps({
        "stage":          "embeddings_log_saved",
        "document_id":    document_id,
        "file_name":      file_name,
        "source_s3_key":  source_s3_key,
        "store_path":     store_path,
        "chunk_count":    len(records),
        "file_size_kb":   round(size_kb, 1),
    }))
    print(
        f"\n📄 Embeddings pkl created → {store_path}  "
        f"(document='{file_name}'  |  {len(records)} chunk(s)  |  {size_kb:.1f} KB)\n"
    )

    if bucket:
        try:
            upload_store_to_s3(store_path, bucket, s3_prefix)
        except Exception as e:
            logger.warning(f"Could not upload pkl '{filename}' to S3 ({e}).")

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


def upload_store_to_s3(store_path: str, bucket: str, s3_prefix: str = STORE_SUBDIR) -> str:
    """
    Upload one document's pkl file to S3 at "{s3_prefix}/{filename}".

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

    logger.info(f"Uploading embeddings pkl → s3://{bucket}/{s3_key}")
    try:
        s3 = boto3.client("s3", verify=False)
        s3.upload_file(store_path, bucket, s3_key)
    except NoCredentialsError as e:
        raise PermissionError(
            "No AWS credentials found while uploading embeddings pkl."
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
        "stage":  "embeddings_pkl_uploaded_to_s3",
        "s3_key": s3_key,
        "bucket": bucket,
    }))
    print(f"☁️  Embeddings pkl uploaded → s3://{bucket}/{s3_key}\n")
    return s3_key


def _purge_pkls_for_key(
    s3_key: str,
    bucket: Optional[str],
    output_dir: str = OUTPUT_DIR,
    s3_prefix: str = STORE_SUBDIR,
) -> Tuple[int, int]:
    """
    Delete every pkl whose filename starts with "{sanitized(s3_key)}__",
    both locally and (if a bucket is given) in S3. Shared by
    save_embeddings_log() (replace-on-reingest) and
    delete_embeddings_by_s3_key() (removal on S3 delete event).

    Returns:
        (files_removed, chunks_removed)
    """
    prefix_match   = f"{_sanitize_key(s3_key)}__"
    store_dir      = _store_dir(output_dir)
    removed_chunks = 0
    matched_files  = 0
    counted_names: set = set()

    # ── Local matches ──────────────────────────────────────────────────────
    try:
        for fname in os.listdir(store_dir):
            if fname.startswith(prefix_match) and fname.endswith(".pkl"):
                local_path = os.path.join(store_dir, fname)
                try:
                    with open(local_path, "rb") as f:
                        data = pickle.load(f)
                    if isinstance(data, list):
                        removed_chunks += len(data)
                except Exception as e:
                    logger.warning(f"Could not read '{fname}' before deleting it ({e}).")
                try:
                    os.remove(local_path)
                    matched_files += 1
                    counted_names.add(fname)
                except OSError as e:
                    logger.warning(f"Could not remove local pkl '{local_path}' ({e}).")
    except FileNotFoundError:
        pass

    # ── S3 matches ────────────────────────────────────────────────────────
    if bucket:
        try:
            s3 = boto3.client("s3", verify=False)
            paginator = s3.get_paginator("list_objects_v2")
            remote_prefix = f"{s3_prefix}/{prefix_match}"
            for page in paginator.paginate(Bucket=bucket, Prefix=remote_prefix):
                for obj in page.get("Contents", []):
                    key   = obj["Key"]
                    fname = key.rsplit("/", 1)[-1]
                    if fname not in counted_names:
                        try:
                            resp = s3.get_object(Bucket=bucket, Key=key)
                            data = pickle.loads(resp["Body"].read())
                            if isinstance(data, list):
                                removed_chunks += len(data)
                        except Exception as e:
                            logger.warning(f"Could not read s3://{bucket}/{key} before deleting it ({e}).")
                    try:
                        s3.delete_object(Bucket=bucket, Key=key)
                        matched_files += 1
                        counted_names.add(fname)
                    except ClientError as e:
                        logger.warning(f"Could not delete s3://{bucket}/{key} ({e}).")
        except (NoCredentialsError, ClientError) as e:
            logger.warning(f"Could not reach S3 to clean up pkls ({e}).")
        except Exception as e:
            # Covers SSO/credential-refresh failures (e.g. UnauthorizedSSOTokenError)
            # and any other unexpected boto3 error — local deletion above still
            # succeeded, so this is logged but non-fatal.
            logger.warning(f"Unexpected error cleaning up S3 pkls for '{s3_key}' ({e}).")

    return matched_files, removed_chunks


def delete_embeddings_by_s3_key(
    s3_key: str,
    bucket: str,
    output_dir: str = OUTPUT_DIR,
    s3_prefix: str = STORE_SUBDIR,
) -> int:

    if not s3_key or not isinstance(s3_key, str):
        raise ValueError(f"delete_embeddings_by_s3_key: invalid s3_key: {s3_key!r}")
    if not bucket or not isinstance(bucket, str):
        raise ValueError(f"delete_embeddings_by_s3_key: invalid bucket: {bucket!r}")

    matched_files, removed_chunks = _purge_pkls_for_key(
        s3_key, bucket=bucket, output_dir=output_dir, s3_prefix=s3_prefix,
    )

    if matched_files == 0:
        logger.warning(json.dumps({
            "stage":   "delete_embeddings_no_match",
            "s3_key":  s3_key,
            "message": "No pkl file(s) matched — nothing to delete.",
        }))
        return 0

    logger.info(json.dumps({
        "stage":          "delete_embeddings_complete",
        "s3_key":         s3_key,
        "files_removed":  matched_files,
        "chunks_removed": removed_chunks,
    }))
    print(f"\n🗑️  Embeddings deleted for '{s3_key}'  ({matched_files} file(s), {removed_chunks} chunk(s))\n")
    return removed_chunks


def get_store_summary(
    output_dir: str = OUTPUT_DIR,
    bucket: Optional[str] = None,
    s3_prefix: str = STORE_SUBDIR,
) -> Dict[str, Any]:
    """
    Build a verification report: how many documents have embeddings stored,
    how many chunks in total, and a per-document breakdown. Reads from S3 if
    a bucket is given (the authoritative view across all Lambda containers),
    otherwise from the local store directory.

    Returns:
        {
            "document_count": int,
            "total_chunks":   int,
            "documents": [
                {
                    "pkl_key":       "embeddings-store/hr__policy.pdf__<uuid>.pkl",
                    "file_name":     "policy.pdf",
                    "document_id":  "<uuid>",
                    "source_s3_key": "hr/policy.pdf",
                    "chunk_count":  12,
                },
                ...
            ],
        }
    """
    documents: List[Dict[str, Any]] = []

    def _summarize(pkl_key: str, raw_bytes: bytes) -> None:
        try:
            data = pickle.loads(raw_bytes)
        except Exception as e:
            logger.warning(f"Could not unpickle '{pkl_key}' for summary ({e}).")
            return
        if not isinstance(data, list) or not data:
            documents.append({
                "pkl_key": pkl_key, "file_name": None, "document_id": None,
                "source_s3_key": None, "chunk_count": 0,
            })
            return
        first = data[0]
        documents.append({
            "pkl_key":       pkl_key,
            "file_name":     first.get("file_name"),
            "document_id":   first.get("document_id"),
            "source_s3_key": first.get("source_s3_key"),
            "chunk_count":   len(data),
        })

    if bucket:
        try:
            s3 = boto3.client("s3", verify=False)
            paginator = s3.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=f"{s3_prefix}/"):
                for obj in page.get("Contents", []):
                    key = obj["Key"]
                    if not key.endswith(".pkl"):
                        continue
                    resp = s3.get_object(Bucket=bucket, Key=key)
                    _summarize(key, resp["Body"].read())
        except Exception as e:
            logger.warning(f"Could not list S3 store for summary ({e}) — falling back to local.")

    if not bucket or not documents:
        store_dir = _store_dir(output_dir)
        for fname in sorted(os.listdir(store_dir)):
            if not fname.endswith(".pkl"):
                continue
            with open(os.path.join(store_dir, fname), "rb") as f:
                _summarize(f"{s3_prefix}/{fname}", f.read())

    return {
        "document_count": len(documents),
        "total_chunks":   sum(d["chunk_count"] for d in documents),
        "documents":      documents,
    }
