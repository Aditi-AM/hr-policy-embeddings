import logging
from typing import List, Dict, Any, Optional

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

# Reciprocal Rank Fusion constant.  k=60 is the standard default that prevents
# top-ranked results from dominating and gives tail results a fair contribution.
_RRF_K = 60


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
                "text_content":   {"type": "text"},   # BM25-analysed for keyword search
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
    if chunks is None:
        raise ValueError("bulk_index_chunks received None instead of a chunk list.")
    if not isinstance(chunks, list):
        raise ValueError(
            f"bulk_index_chunks expected a list, got {type(chunks).__name__}."
        )
    if not chunks:
        logger.warning("bulk_index_chunks called with an empty list — nothing to index.")
        return 0

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

    Args:
        s3_key: The S3 object key of the deleted document (e.g. "hr/policy.pdf").

    Returns:
        Number of documents deleted from the index.
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
            "term": {"source_s3_key": s3_key}
        }
    }

    logger.info(
        f"Running delete_by_query on index '{index}' for source_s3_key='{s3_key}'"
    )

    try:
        response = client.delete_by_query(
            index=index,
            body=query,
            conflicts="proceed",
            wait_for_completion=True,
        )
    except AuthenticationException as e:
        raise PermissionError(
            f"OpenSearch authentication failed during delete_by_query: {e}"
        ) from e
    except AuthorizationException as e:
        raise PermissionError(
            f"OpenSearch access denied for delete_by_query on '{index}': {e}."
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


def search_similar_chunks(
    query_vector: List[float],
    query_text: str = "",
    top_k: int = 5,
    department: Optional[str] = None,
    policy_type: Optional[str] = None,
) -> List[Dict[str, Any]]:

    if not isinstance(query_vector, list) or not query_vector:
        raise ValueError("search_similar_chunks: query_vector must be a non-empty list.")
    if top_k < 1:
        raise ValueError(f"search_similar_chunks: top_k must be >= 1, got {top_k}.")

    # Pure vector search when no query text is provided
    if not query_text or not query_text.strip():
        logger.info(
            f"[opensearch] search_similar_chunks: no query_text — "
            f"falling back to pure vector search (top_k={top_k})"
        )
        return _vector_search(query_vector, top_k, department, policy_type)

    # Hybrid: fetch a larger candidate set from each retriever, then merge with RRF
    fetch_k = min(top_k * 3, 30)   # cap at 30 to keep Lambda latency reasonable

    logger.info(
        f"[opensearch] search_similar_chunks: hybrid mode — "
        f"top_k={top_k} fetch_k={fetch_k} "
        f"department={department!r} policy_type={policy_type!r}"
    )

    vector_hits  = _vector_search(query_vector, fetch_k, department, policy_type)
    keyword_hits = _keyword_search(query_text.strip(), fetch_k, department, policy_type)

    merged = _rrf_merge(vector_hits, keyword_hits, top_k)

    logger.info(
        f"[opensearch] search_similar_chunks: RRF merge — "
        f"vector={len(vector_hits)} keyword={len(keyword_hits)} "
        f"merged={len(merged)}"
    )
    return merged



def _build_filters(department: Optional[str], policy_type: Optional[str]) -> List[dict]:
    """Return a list of term-filter clauses for metadata narrowing."""
    filters: List[dict] = []
    if department:
        filters.append({"term": {"department": department}})
    if policy_type:
        filters.append({"term": {"policy_type": policy_type}})
    return filters


def _vector_search(
    query_vector: List[float],
    top_k: int,
    department: Optional[str] = None,
    policy_type: Optional[str] = None,
) -> List[Dict[str, Any]]:

    config = get_config()
    index  = config["opensearch_index"]

    try:
        client = _get_client()
    except RuntimeError:
        raise

    knn_clause = {
        "knn": {
            "embedding": {
                "vector": query_vector,
                "k": top_k,
            }
        }
    }

    filters = _build_filters(department, policy_type)
    if filters:
        query_body = {
            "bool": {
                "must":   [knn_clause],
                "filter": filters,
            }
        }
    else:
        query_body = knn_clause

    search_payload = {
        "size": top_k,
        "query": query_body,
        "_source": {"excludes": ["embedding"]},
    }

    logger.info(
        f"[opensearch] _vector_search: index={index} top_k={top_k} "
        f"department={department!r} policy_type={policy_type!r}"
    )

    try:
        response = client.search(index=index, body=search_payload)
    except AuthenticationException as e:
        raise PermissionError(
            f"OpenSearch authentication failed during vector search: {e}"
        ) from e
    except AuthorizationException as e:
        raise PermissionError(
            f"OpenSearch access denied for vector search on '{index}': {e}"
        ) from e
    except (OSConnectionError, ConnectionTimeout) as e:
        raise ConnectionError(
            f"Cannot reach OpenSearch endpoint during vector search: {e}"
        ) from e
    except TransportError as e:
        raise RuntimeError(
            f"OpenSearch transport error during vector search on '{index}': "
            f"status={e.status_code}, error={e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error during vector search on '{index}': {e}"
        ) from e

    hits = response.get("hits", {}).get("hits", [])
    results: List[Dict[str, Any]] = []
    for hit in hits:
        doc = hit.get("_source", {})
        doc["_score"] = hit.get("_score")
        doc["_id"]    = hit.get("_id")
        results.append(doc)

    logger.info(
        f"[opensearch] _vector_search: {len(results)} hit(s) "
        f"(top score={results[0]['_score'] if results else 'n/a'})"
    )
    return results


def _keyword_search(
    query_text: str,
    top_k: int,
    department: Optional[str] = None,
    policy_type: Optional[str] = None,
) -> List[Dict[str, Any]]:

    config = get_config()
    index  = config["opensearch_index"]

    try:
        client = _get_client()
    except RuntimeError:
        raise

    match_clause = {
        "multi_match": {
            "query":                query_text,
            "fields":               ["text_content"],
            "type":                 "best_fields",
            "operator":             "or",
            # For 1–2 word queries all terms must appear;
            # for 3+ word queries at least 75 % must appear.
            "minimum_should_match": "2<75%",
        }
    }

    filters = _build_filters(department, policy_type)
    if filters:
        query_body = {
            "bool": {
                "must":   [match_clause],
                "filter": filters,
            }
        }
    else:
        query_body = match_clause

    search_payload = {
        "size": top_k,
        "query": query_body,
        "_source": {"excludes": ["embedding"]},
    }

    logger.info(
        f"[opensearch] _keyword_search: index={index} top_k={top_k} "
        f"query='{query_text[:80]}' "
        f"department={department!r} policy_type={policy_type!r}"
    )

    try:
        response = client.search(index=index, body=search_payload)
    except AuthenticationException as e:
        raise PermissionError(
            f"OpenSearch authentication failed during keyword search: {e}"
        ) from e
    except AuthorizationException as e:
        raise PermissionError(
            f"OpenSearch access denied for keyword search on '{index}': {e}"
        ) from e
    except (OSConnectionError, ConnectionTimeout) as e:
        raise ConnectionError(
            f"Cannot reach OpenSearch endpoint during keyword search: {e}"
        ) from e
    except TransportError as e:
        raise RuntimeError(
            f"OpenSearch transport error during keyword search on '{index}': "
            f"status={e.status_code}, error={e}"
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"Unexpected error during keyword search on '{index}': {e}"
        ) from e

    hits = response.get("hits", {}).get("hits", [])
    results: List[Dict[str, Any]] = []
    for hit in hits:
        doc = hit.get("_source", {})
        doc["_score"] = hit.get("_score")
        doc["_id"]    = hit.get("_id")
        results.append(doc)

    logger.info(
        f"[opensearch] _keyword_search: {len(results)} hit(s) "
        f"(top score={results[0]['_score'] if results else 'n/a'})"
    )
    return results


def _rrf_merge(
    vector_hits: List[Dict[str, Any]],
    keyword_hits: List[Dict[str, Any]],
    top_k: int,
    k: int = _RRF_K,
) -> List[Dict[str, Any]]:

    # doc_id → source dict (first occurrence wins; both lists share the same _source)
    docs: Dict[str, Dict[str, Any]] = {}

    for hit in vector_hits + keyword_hits:
        # chunk_id is in _source and equals the OpenSearch _id
        doc_id = hit.get("chunk_id") or hit.get("_id")
        if doc_id and doc_id not in docs:
            docs[doc_id] = hit

    if not docs:
        return []

    # Accumulate RRF scores
    rrf_scores: Dict[str, float] = {doc_id: 0.0 for doc_id in docs}

    for rank, hit in enumerate(vector_hits, start=1):
        doc_id = hit.get("chunk_id") or hit.get("_id")
        if doc_id in rrf_scores:
            rrf_scores[doc_id] += 1.0 / (k + rank)

    for rank, hit in enumerate(keyword_hits, start=1):
        doc_id = hit.get("chunk_id") or hit.get("_id")
        if doc_id in rrf_scores:
            rrf_scores[doc_id] += 1.0 / (k + rank)

    # Sort by combined score, return top_k
    sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)

    results: List[Dict[str, Any]] = []
    for doc_id in sorted_ids[:top_k]:
        doc = {k_: v for k_, v in docs[doc_id].items() if k_ not in ("_score", "_id")}
        doc["_score"] = round(rrf_scores[doc_id], 6)
        results.append(doc)

    return results
