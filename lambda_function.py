
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import unquote_plus

from s3_utils import download_document, get_object_tags, cleanup_tmp
from file_handler import process_document
from ai_gateway_utils import embed_chunks
from opensearch_utils import ensure_index, bulk_index_chunks, delete_chunks_by_s3_key
from embeddings_log_utils import save_embeddings_log, upload_store_to_s3, delete_embeddings_by_s3_key

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def lambda_handler(event, context):
    document_id = str(uuid.uuid4())
    local_path  = None

    # ── Step 0: Parse S3 event ────────────────────────────────────────────────
    try:
        records = event.get("Records")
        if not records or not isinstance(records, list):
            raise ValueError(f"Event missing 'Records'. Received: {json.dumps(event)}")

        record     = records[0]
        event_name = record.get("eventName", "")
        bucket     = record["s3"]["bucket"]["name"]
        key        = unquote_plus(record["s3"]["object"]["key"])  # S3 event keys are URL-encoded

        if not bucket or not key:
            raise ValueError("Blank bucket or key in event.")
    except (KeyError, TypeError) as e:
        _log_error("event_parse_error", document_id, e,
                   hint="Verify the Lambda trigger is an S3 event (ObjectCreated or ObjectRemoved).")
        raise ValueError(f"Malformed S3 event: {e}") from e

    _log_info("start", document_id, bucket=bucket, key=key, event_name=event_name)

    # ── Skip non-document files (e.g. embeddings PKL fallback uploads) ────────
    _SUPPORTED_EXTENSIONS = {"pdf", "docx", "doc"}
    file_name  = key.split("/")[-1]
    extension  = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
    if extension not in _SUPPORTED_EXTENSIONS:
        logger.info(json.dumps({
            "stage":      "unsupported_file_skipped",
            "document_id": document_id,
            "key":        key,
            "extension":  extension,
            "hint":       f"Only {sorted(_SUPPORTED_EXTENSIONS)} files are processed.",
        }))
        return {
            "statusCode": 200,
            "body": json.dumps({
                "document_id": document_id,
                "key":         key,
                "result":      "skipped — unsupported file type",
            }),
        }

    # ── Route by event type ───────────────────────────────────────────────────
    if event_name.startswith("ObjectRemoved"):
        return _handle_delete(document_id, bucket, key)

    if event_name.startswith("ObjectCreated"):
        try:
            return _handle_ingest(document_id, bucket, key, local_path)
        finally:
            if local_path and os.path.exists(local_path):
                try:
                    cleanup_tmp(local_path)
                    logger.info(json.dumps({"stage": "cleanup_done", "local_path": local_path}))
                except Exception as e:
                    logger.warning(json.dumps({"stage": "cleanup_failed", "error": str(e)}))

    # Unknown event type — log and skip gracefully
    logger.warning(json.dumps({
        "stage":      "unknown_event_skipped",
        "event_name": event_name,
        "key":        key,
        "hint":       "Only ObjectCreated and ObjectRemoved events are handled.",
    }))
    return {
        "statusCode": 200,
        "body": json.dumps({
            "document_id": document_id,
            "event_name":  event_name,
            "result":      "skipped — unrecognised event type",
        }),
    }


# ── Ingest handler (ObjectCreated:Put / ObjectCreated:Copy / etc.) ─────────────

def _handle_ingest(document_id: str, bucket: str, key: str, local_path) -> dict:
    ingested_at = datetime.now(timezone.utc).isoformat()
    file_name   = key.split("/")[-1]

    try:
        # ── Step 1: Download from S3 ──────────────────────────────────────────
        try:
            local_path = download_document(bucket, key)
        except Exception as e:
            _log_error("s3_download_error", document_id, e)
            raise

        _log_info("s3_download_complete", document_id, local_path=local_path)

        # ── Step 2: Extract metadata tags ─────────────────────────────────────
        try:
            metadata = get_object_tags(bucket, key)
            metadata.update({
                "document_id":   document_id,
                "source_s3_key": key,
                "file_name":     file_name,
                "ingested_at":   ingested_at,
            })
        except Exception as e:
            _log_error("metadata_error", document_id, e)
            raise

        _log_info("metadata_extracted", document_id, metadata=metadata)

        # ── Step 3: Parse document into chunks ────────────────────────────────
        try:
            chunks = process_document(local_path, metadata)
        except Exception as e:
            _log_error("parsing_error", document_id, e)
            raise

        if not chunks:
            raise RuntimeError(f"Parsing produced 0 chunks for key={key}.")

        _log_info("chunks_created", document_id, chunk_count=len(chunks))

        # ── Step 4: Embed chunks via AI Gateway ───────────────────────────────
        try:
            chunks_with_embeddings = embed_chunks(chunks)
        except Exception as e:
            _log_error("embedding_fatal_error", document_id, e)
            raise

        _log_info("embeddings_generated", document_id,
                  embedded_count=len(chunks_with_embeddings),
                  skipped=len(chunks) - len(chunks_with_embeddings))

        if not chunks_with_embeddings:
            raise RuntimeError(
                f"All {len(chunks)} chunk(s) failed embedding for "
                f"document_id={document_id}. Nothing to index."
            )

        # ── Step 4b: Save embeddings log to PKL (non-fatal) ───────────────────
        log_path = None
        try:
            log_path = save_embeddings_log(chunks_with_embeddings, document_id)
            _log_info("embeddings_log_saved", document_id, log_path=log_path)
        except Exception as e:
            logger.warning(json.dumps({
                "stage":       "embeddings_log_failed",
                "document_id": document_id,
                "error":       str(e),
                "hint":        "PKL log not written — indexing will continue.",
            }))

        # ── Step 5: Index to OpenSearch (fall back to S3 pkl if unavailable) ──
        indexed_count        = 0
        store_s3_key         = None
        opensearch_available = True

        try:
            ensure_index()
            indexed_count = bulk_index_chunks(chunks_with_embeddings)
            _log_info("indexing_complete", document_id, indexed_count=indexed_count)
        except Exception as e:
            opensearch_available = False
            _log_error("opensearch_error", document_id, e,
                       hint="OpenSearch unavailable — falling back to S3 pkl.")

        if not opensearch_available and log_path:
            try:
                store_s3_key = upload_store_to_s3(log_path, bucket)
                _log_info("embeddings_store_uploaded", document_id, s3_key=store_s3_key)
            except Exception as e:
                _log_error("embeddings_store_upload_failed", document_id, e)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "action":         "ingest",
                "document_id":    document_id,
                "chunks_total":   len(chunks),
                "embedded_count": len(chunks_with_embeddings),
                "chunks_indexed": indexed_count,
                "store_s3_key":   store_s3_key,
            }),
        }

    finally:
        # Always clean up the downloaded file from /tmp
        if local_path and os.path.exists(local_path):
            try:
                cleanup_tmp(local_path)
                logger.info(json.dumps({"stage": "cleanup_done", "local_path": local_path}))
            except Exception as e:
                logger.warning(json.dumps({"stage": "cleanup_failed", "error": str(e)}))


# ── Delete handler (ObjectRemoved:Delete / ObjectRemoved:DeleteMarkerCreated) ──

def _handle_delete(document_id: str, bucket: str, key: str) -> dict:
    _log_info("delete_start", document_id, bucket=bucket, key=key)

    opensearch_deleted = 0
    pkl_deleted        = 0

    # ── Step 1: Remove from OpenSearch ────────────────────────────────────────
    try:
        opensearch_deleted = delete_chunks_by_s3_key(key)
        _log_info("opensearch_delete_complete", document_id,
                  key=key, chunks_deleted=opensearch_deleted)
    except Exception as e:
        _log_error("opensearch_delete_error", document_id, e,
                   hint="OpenSearch delete failed — PKL deletion will still be attempted.")

    # ── Step 2: Remove from PKL store ─────────────────────────────────────────
    try:
        pkl_deleted = delete_embeddings_by_s3_key(s3_key=key, bucket=bucket)
        _log_info("pkl_delete_complete", document_id,
                  key=key, chunks_deleted=pkl_deleted)
    except RuntimeError as e:
        # Store file not found — nothing to clean up in PKL, not a hard error
        logger.warning(json.dumps({
            "stage":       "pkl_delete_skipped",
            "document_id": document_id,
            "key":         key,
            "error":       str(e),
            "hint":        "PKL store may not exist yet — skipping.",
        }))
    except Exception as e:
        _log_error("pkl_delete_error", document_id, e,
                   hint="PKL deletion failed — manual cleanup may be needed.")

    _log_info("delete_complete", document_id, key=key,
              opensearch_deleted=opensearch_deleted, pkl_deleted=pkl_deleted)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "action":              "delete",
            "document_id":         document_id,
            "s3_key":              key,
            "opensearch_deleted":  opensearch_deleted,
            "pkl_deleted":         pkl_deleted,
        }),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_info(stage: str, document_id: str, **kwargs) -> None:
    logger.info(json.dumps({"stage": stage, "document_id": document_id, **kwargs}))


def _log_error(stage: str, document_id: str, error: Exception, hint: str = "") -> None:
    payload = {
        "stage":       stage,
        "document_id": document_id,
        "error_type":  type(error).__name__,
        "error":       str(error),
    }
    if hint:
        payload["hint"] = hint
    logger.error(json.dumps(payload))
