
import os
import json
import logging
import traceback
import boto3
import requests
from botocore.exceptions import ClientError
from typing import List, Dict, Any

from config_utils import get_config

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = int(os.environ.get("AI_GATEWAY_TIMEOUT", "30"))

_api_key: str = None


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------

def _resolve_api_key() -> str:
    """Fetch the AI Gateway API key from Secrets Manager."""
    logger.info("[ai_gateway] _resolve_api_key: loading config from ASM")
    try:
        config = get_config()
    except Exception as e:
        logger.error(
            f"[ai_gateway] _resolve_api_key: get_config() failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise

    try:
        secret_name = config.get("ai_gateway_token_secret_name", "").strip()
        asm_region  = config.get("asm_region", "ap-south-1")

        logger.info(
            f"[ai_gateway] _resolve_api_key: fetching token from "
            f"secret='{secret_name}' region='{asm_region}'"
        )

        if not secret_name:
            raise RuntimeError(
                "Config secret is missing 'ai_gateway_token_secret_name'. "
                "Update hr-bot-config-dev in ASM."
            )
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(
            f"[ai_gateway] _resolve_api_key: failed to parse config — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to parse AI Gateway config: {e}") from e

    try:
        client   = boto3.client("secretsmanager", region_name=asm_region)
        response = client.get_secret_value(SecretId=secret_name)
        logger.info(f"[ai_gateway] _resolve_api_key: successfully fetched secret '{secret_name}'")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        logger.error(
            f"[ai_gateway] _resolve_api_key: ClientError code='{code}' "
            f"secret='{secret_name}' — {e}\n{traceback.format_exc()}"
        )
        if code == "ResourceNotFoundException":
            raise RuntimeError(f"AI Gateway token secret '{secret_name}' not found in ASM.") from e
        elif code in ("AccessDeniedException", "AccessDenied"):
            raise PermissionError(
                f"Access denied to secret '{secret_name}'. "
                "Add secretsmanager:GetSecretValue to the Lambda execution role."
            ) from e
        else:
            raise RuntimeError(f"ASM ClientError for secret '{secret_name}': {e}") from e
    except Exception as e:
        logger.error(
            f"[ai_gateway] _resolve_api_key: unexpected error fetching secret — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to fetch AI Gateway token: {e}") from e

    try:
        secret_string = response.get("SecretString", "")
        if not secret_string:
            raise RuntimeError(f"Secret '{secret_name}' has no SecretString.")

        api_key = json.loads(secret_string).get("api_key_value", "").strip()

        if not api_key:
            raise RuntimeError(f"Secret '{secret_name}' has no 'api_key_value' field.")
    except json.JSONDecodeError as e:
        logger.error(
            f"[ai_gateway] _resolve_api_key: secret '{secret_name}' is not valid JSON — "
            f"{e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Secret '{secret_name}' is not valid JSON: {e}") from e
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(
            f"[ai_gateway] _resolve_api_key: failed to parse secret response — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to parse AI Gateway secret: {e}") from e

    logger.info(f"[ai_gateway] _resolve_api_key: token resolved (length={len(api_key)})")
    return api_key


def _get_api_key() -> str:
    """Return the cached API key, resolving it on first call."""
    global _api_key
    try:
        if _api_key is None:
            logger.info("[ai_gateway] _get_api_key: cache miss — resolving token")
            _api_key = _resolve_api_key()
        else:
            logger.info("[ai_gateway] _get_api_key: using cached token")
        return _api_key
    except Exception as e:
        logger.error(
            f"[ai_gateway] _get_api_key: failed to resolve API key — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Embed a list of text chunks via the AI Gateway. Returns chunks with embeddings."""
    logger.info(f"[ai_gateway] embed_chunks: received {len(chunks) if chunks else 0} chunks")

    if not chunks:
        logger.warning("[ai_gateway] embed_chunks: empty chunk list — returning []")
        return []

    results = []
    failed  = 0

    for i, chunk in enumerate(chunks):
        chunk_id = chunk.get("chunk_id", f"idx-{i}")
        logger.info(f"[ai_gateway] embed_chunks: processing chunk {i+1}/{len(chunks)} chunk_id={chunk_id}")
        try:
            vector = _embed_single_chunk(chunk)
            chunk["embedding"] = vector
            results.append(chunk)
            logger.info(f"[ai_gateway] embed_chunks: chunk {chunk_id} embedded OK dim={len(vector)}")

        except PermissionError as e:
            logger.critical(
                f"[ai_gateway] embed_chunks: AUTH FAILURE chunk_id={chunk_id} — "
                f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
            )
            raise RuntimeError("AI Gateway authentication failed — aborting embedding.") from e

        except requests.exceptions.Timeout as e:
            failed += 1
            logger.error(
                f"[ai_gateway] embed_chunks: TIMEOUT chunk_id={chunk_id} — "
                f"error={e}\n{traceback.format_exc()}"
            )

        except requests.exceptions.ConnectionError as e:
            failed += 1
            logger.error(
                f"[ai_gateway] embed_chunks: CONNECTION ERROR chunk_id={chunk_id} — "
                f"error={e}\n{traceback.format_exc()}"
            )

        except requests.exceptions.HTTPError as e:
            failed += 1
            status = e.response.status_code if e.response is not None else "unknown"
            body   = e.response.text[:300] if e.response is not None else ""
            logger.error(
                f"[ai_gateway] embed_chunks: HTTP ERROR chunk_id={chunk_id} "
                f"status={status} body={body} — {e}\n{traceback.format_exc()}"
            )

        except ValueError as e:
            failed += 1
            logger.warning(
                f"[ai_gateway] embed_chunks: SKIPPED chunk_id={chunk_id} "
                f"reason={e}\n{traceback.format_exc()}"
            )

        except Exception as e:
            failed += 1
            logger.error(
                f"[ai_gateway] embed_chunks: UNEXPECTED ERROR chunk_id={chunk_id} "
                f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
            )

    logger.info(
        f"[ai_gateway] embed_chunks: done — {len(results)} OK, "
        f"{failed} failed out of {len(chunks)}"
    )
    return results


def _embed_single_chunk(chunk: Dict[str, Any]) -> List[float]:
    """Call the AI Gateway to embed a single text chunk."""
    chunk_id = chunk.get("chunk_id", "unknown")
    text     = chunk.get("text_content", "")

    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"chunk_id={chunk_id} has empty or invalid text_content.")
    text = text.strip()

    logger.info(f"[ai_gateway] _embed_single_chunk: chunk_id={chunk_id} text_length={len(text)}")

    # Resolve config and token
    try:
        api_key = _get_api_key()
    except Exception as e:
        logger.error(
            f"[ai_gateway] _embed_single_chunk: failed to get api_key — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise

    try:
        config = get_config()
    except Exception as e:
        logger.error(
            f"[ai_gateway] _embed_single_chunk: failed to get config — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise

    try:
        url   = f"{config['ai_gateway_url'].rstrip('/')}/embeddings"
        model = config["embed_model"]
    except KeyError as e:
        logger.error(
            f"[ai_gateway] _embed_single_chunk: missing config key {e} — "
            f"available keys={list(config.keys())}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Config is missing required key: {e}") from e

    logger.info(
        f"[ai_gateway] _embed_single_chunk: POST {url} model={model} "
        f"timeout={REQUEST_TIMEOUT}s"
    )

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={"model": model, "input": text},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout as e:
        logger.error(
            f"[ai_gateway] _embed_single_chunk: request timed out after "
            f"{REQUEST_TIMEOUT}s url={url}\n{traceback.format_exc()}"
        )
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(
            f"[ai_gateway] _embed_single_chunk: cannot connect to {url} — "
            f"error={e}\n{traceback.format_exc()}"
        )
        raise
    except Exception as e:
        logger.error(
            f"[ai_gateway] _embed_single_chunk: requests.post failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise

    logger.info(
        f"[ai_gateway] _embed_single_chunk: response status={response.status_code} "
        f"chunk_id={chunk_id}"
    )

    if response.status_code in (401, 403):
        logger.error(
            f"[ai_gateway] _embed_single_chunk: auth rejected "
            f"status={response.status_code} body={response.text[:300]}"
        )
        raise PermissionError(
            f"AI Gateway returned {response.status_code} for "
            f"chunk_id={chunk_id}: {response.text[:200]}"
        )

    if not response.ok:
        logger.error(
            f"[ai_gateway] _embed_single_chunk: non-2xx "
            f"status={response.status_code} body={response.text[:300]}"
        )
        response.raise_for_status()

    try:
        body      = response.json()
        embedding = body["data"][0]["embedding"]
    except (KeyError, IndexError, ValueError) as e:
        logger.error(
            f"[ai_gateway] _embed_single_chunk: unexpected response shape — "
            f"type={type(e).__name__} error={e} body={response.text[:300]}\n"
            f"{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"Unexpected AI Gateway response shape: {e}. "
            f"Body: {response.text[:200]}"
        ) from e
    except Exception as e:
        logger.error(
            f"[ai_gateway] _embed_single_chunk: failed to parse response — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to parse AI Gateway response: {e}") from e

    if not isinstance(embedding, list) or not embedding:
        raise RuntimeError(f"AI Gateway returned empty embedding for chunk_id={chunk_id}.")

    logger.info(
        f"[ai_gateway] _embed_single_chunk: SUCCESS chunk_id={chunk_id} "
        f"dim={len(embedding)} model={body.get('model')} "
        f"tokens={body.get('usage', {}).get('total_tokens')}"
    )
    return embedding
