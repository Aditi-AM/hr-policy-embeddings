

import logging
from typing import List, Dict, Any

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers
from opensearchpy.exceptions import (
    RequestError,
    ConnectionError as OSConnectionError,
    ConnectionTimeout,
    TransportError,
    AuthenticationException,
    AuthorizationException,
)

from config_utils import get_config

logger = logging.getLogger(__name__)

# Config is loaded lazily from ASM via get_config() — no env vars needed.


def _get_index_body(embedding_dimension: int) -> dict:
    """Build the index mapping using the embedding dimension from config."""
    return {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512
            }
        },
        "mappings": {
            "properties": {
                "document_id":    {"type": "keyword"},
                "chunk_id":       {"type": "keyword"},
                "source_s3_key":  {"type": "keyword"},
                "file_name":      {"type": "keyword"},
                "file_type":      {"type": "keyword"},
                "page_number":    {"type": "integer"},
                "chunk_index":    {"type": "integer"},
                "text_content":   {"type": "text"},
                "department":     {"type": "keyword"},
                "policy_type":    {"type": "keyword"},
                "effective_date": {"type": "date"},
                "ingested_at":    {"type": "date"},
                "embedding": {
                    "type":      "knn_vector",
                    "dimension": embedding_dimension,
                    "method": {
                        "name":       "hnsw",
                        "engine":     "nmslib",
                        "space_type": "cosinesimil"
                    }
                }
            }
        }
    }


def _get_client() -> OpenSearch:
    """
    Build and return an OpenSearch client using endpoint from ASM config.

    Raises:
        RuntimeError: Config not loaded or client could not be constructed.
    """
    config   = get_config()
    endpoint = config.get("opensearch_endpoint", "").strip()

    if not endpoint:
        raise RuntimeError(
            "Config secret is missing 'opensearch_endpoint'. "
            "Update hr-bot-config-dev in ASM."
        )

    # Strip https:// — the client adds it via use_ssl=True
    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")

    try:
        return OpenSearch(
            hosts=[{"host": host, "port": 443}],
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            timeout=30,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create OpenSearch client for endpoint '{endpoint}': {e}"
        ) from e


def ensure_index() -> None:
    """
    Create the OpenSearch k-NN index if it does not already exist.
    Safe to call on every Lambda invocation — no-op if index exists.
    """
    config    = get_config()
    index     = config["opensearch_index"]
    emb_dim   = int(config["embedding_dimension"])
    index_body = _get_index_body(emb_dim)

    try:
        client = _get_client()
    except RuntimeError:
        raise

    # --- Check existence ---
    try:
        exists = client.indices.exists(index=index)
    except AuthenticationException as e:
        raise PermissionError(
            f"OpenSearch authentication failed while checking index '{index}': {e}"
        ) from e
    except AuthorizationException as e:
        raise PermissionError(
            f"OpenSearch access denied for index '{index}': {e}. "
            "Check IAM / fine-grained access control permissions."
        ) from e
    except (OSConnectionError, ConnectionTimeout) as e:
        raise ConnectionError(
            f"Cannot reach OpenSearch endpoint: {e}"
        ) from e
    except TransportError as e:
        raise RuntimeError(
            f"OpenSearch transport error checking index '{index}': "
            f"status={e.status_code}, error={e}"
        ) from e

    if exists:
        logger.info(f"Index '{index}' already exists — skipping creation.")
        return

    # --- Create index ---
    try:
        client.indices.create(index=index, body=index_body)
        logger.info(f"Index '{index}' created successfully.")
    except RequestError as e:
        if "resource_already_exists_exception" in str(e).lower():
            logger.info(f"Index '{index}' was created concurrently — continuing.")
        else:
            raise RuntimeError(f"Failed to create index '{index}': {e}") from e
    except AuthorizationException as e:
        raise PermissionError(f"Not authorised to create index '{index}': {e}") from e
    except (OSConnectionError, ConnectionTimeout) as e:
        raise ConnectionError(
            f"Lost connection to OpenSearch while creating index '{index}': {e}"
        ) from e
    except TransportError as e:
        raise RuntimeError(
            f"OpenSearch transport error creating index '{index}': "
            f"status={e.status_code}, error={e}"
        ) from e


def bulk_index_chunks(chunks: List[Dict[str, Any]]) -> int:
    """
    Bulk index all chunks into the OpenSearch k-NN index.

    Args:
        chunks: List of chunk dicts — each must have 'chunk_id' and 'embedding'.

    Returns:
        Count of successfully indexed chunks.

    Raises:
        ValueError:       chunks is None, not a list, or chunks are malformed.
        PermissionError:  Authentication or authorisation failure.
        ConnectionError:  Cannot reach the OpenSearch cluster.
        RuntimeError:     Bulk indexing failed entirely.
    """
    # --- Input validation ---
    if chunks is None:
        raise ValueError("bulk_index_chunks received None instead of a chunk list.")
    if not isinstance(chunks, list):
        raise ValueError(
            f"bulk_index_chunks expected a list, got {type(chunks).__name__}."
        )
    if not chunks:
        logger.warning("bulk_index_chunks called with an empty list — nothing to index.")
        return 0

    # Validate each chunk has required fields
    invalid = [
        i for i, c in enumerate(chunks)
        if not isinstance(c, dict) or not c.get("chunk_id") or "embedding" not in c
    ]
    if invalid:
        raise ValueError(
            f"Chunks at indices {invalid} are missing 'chunk_id' or 'embedding'. "
            "Ensure embed_chunks() ran successfully before indexing."
        )

    config = get_config()
    index  = config["opensearch_index"]

    try:
        client = _get_client()
    except (ValueError, RuntimeError):
        raise

    actions = []
    for chunk in chunks:
        try:
            actions.append(_build_action(chunk, index))
        except Exception as e:
            logger.error(
                f"Failed to build action for chunk_id={chunk.get('chunk_id')}: {e} — skipping."
            )

    if not actions:
        raise RuntimeError(
            "No valid bulk actions could be built from the provided chunks."
        )

    # --- Bulk index ---
    try:
        success_count, errors = helpers.bulk(
            client,
            actions,
            raise_on_error=False,
            stats_only=False,
        )
    except AuthenticationException as e:
        raise PermissionError(
            f"OpenSearch authentication failed during bulk indexing: {e}"
        ) from e
    except AuthorizationException as e:
        raise PermissionError(
            f"OpenSearch access denied during bulk indexing to '{index}': {e}"
        ) from e
    except (OSConnectionError, ConnectionTimeout) as e:
        raise ConnectionError(
            f"Connection to OpenSearch lost during bulk indexing: {e}"
        ) from e
    except TransportError as e:
        raise RuntimeError(
            f"OpenSearch transport error during bulk indexing: "
            f"status={e.status_code}, error={e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error during bulk indexing: {e}") from e

    # --- Log per-document errors (partial failures) ---
    if errors:
        for err in errors:
            logger.error({"stage": "indexing_partial_error", "error": err})
        logger.warning(
            f"Bulk indexing partial failure: {len(errors)} chunk(s) failed to index."
        )

    logger.info(
        f"Bulk indexing complete: {success_count} succeeded, "
        f"{len(errors)} failed out of {len(actions)} actions."
    )
    return success_count


def _build_action(chunk: Dict[str, Any], index: str) -> dict:
    """
    Build a single opensearch-py bulk action dict for a chunk.

    Excludes 'image_path' — it is a local /tmp path irrelevant to the index.

    Args:
        chunk: Chunk dict with all metadata and embedding fields.

    Returns:
        opensearch-py bulk action dict.

    Raises:
        ValueError: chunk_id or embedding is missing/invalid.
    """
    if not chunk.get("chunk_id"):
        raise ValueError("chunk is missing 'chunk_id' — cannot build index action.")

    embedding = chunk.get("embedding")
    if not isinstance(embedding, list) or not embedding:
        raise ValueError(
            f"chunk_id={chunk['chunk_id']} has an invalid or empty 'embedding' field."
        )

    doc = {k: v for k, v in chunk.items() if k != "image_path"}

    return {
        "_index":  index,
        "_id":     chunk["chunk_id"],
        "_source": doc,
    }


def delete_chunks_by_s3_key(s3_key: str) -> int:
    """
    Delete all OpenSearch documents whose source_s3_key matches s3_key.

    Uses the Delete By Query API so no individual chunk IDs are needed —
    the s3_key is the only identifier required.

    Args:
        s3_key: The S3 object key of the deleted document (e.g. "hr/policy.pdf").

    Returns:
        Number of documents deleted from the index.

    Raises:
        ValueError:      s3_key is blank.
        PermissionError: Authentication or authorisation failure.
        ConnectionError: Cannot reach the OpenSearch cluster.
        RuntimeError:    Delete-by-query failed or returned errors.
    """
    if not s3_key or not isinstance(s3_key, str):
        raise ValueError(f"delete_chunks_by_s3_key: invalid s3_key: {s3_key!r}")

    config = get_config()
    index  = config["opensearch_index"]

    try:
        client = _get_client()
    except RuntimeError:
        raise

    query = {
        "query": {
            "term": {
                "source_s3_key": s3_key
            }
        }
    }

    logger.info(f"Running delete_by_query on index '{index}' for source_s3_key='{s3_key}'")

    try:
        response = client.delete_by_query(
            index=index,
            body=query,
            conflicts="proceed",   # skip version conflicts, don't abort
            wait_for_completion=True,
        )
    except AuthenticationException as e:
        raise PermissionError(
            f"OpenSearch authentication failed during delete_by_query: {e}"
        ) from e
    except AuthorizationException as e:
        raise PermissionError(
            f"OpenSearch access denied for delete_by_query on '{index}': {e}. "
            "Check IAM / fine-grained access control permissions."
        ) from e
    except (OSConnectionError, ConnectionTimeout) as e:
        raise ConnectionError(
            f"Cannot reach OpenSearch endpoint during delete_by_query: {e}"
        ) from e
    except TransportError as e:
        raise RuntimeError(
            f"OpenSearch transport error during delete_by_query on '{index}': "
            f"status={e.status_code}, error={e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error during delete_by_query for s3_key='{s3_key}': {e}"
        ) from e

    # ── Parse response ───────────────────────────────────────────────────────
    deleted  = response.get("deleted", 0)
    failures = response.get("failures", [])

    if failures:
        for failure in failures:
            logger.error({"stage": "delete_by_query_failure", "detail": failure})
        logger.warning(
            f"delete_by_query partial failure: {len(failures)} shard failure(s) "
            f"for s3_key='{s3_key}'."
        )

    logger.info(
        f"delete_by_query complete: {deleted} document(s) removed from '{index}' "
        f"for source_s3_key='{s3_key}'."
    )
    return deleted
