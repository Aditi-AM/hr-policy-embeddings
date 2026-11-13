
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

    results        = []
    failure_reasons: List[str] = []  # collect unique failure reasons for diagnostics

    for i, chunk in enumerate(chunks):
        chunk_id  = chunk.get("chunk_id", f"idx-{i}")
        file_name = chunk.get("file_name", "unknown")
        logger.info(f"[ai_gateway] embed_chunks: processing chunk {i+1}/{len(chunks)} chunk_id={chunk_id} file_name={file_name}")
        try:
            vector = _embed_single_chunk(chunk)
            chunk["embedding"] = vector
            results.append(chunk)
            logger.info(f"[ai_gateway] embed_chunks: chunk {chunk_id} embedded OK dim={len(vector)} file_name={file_name}")

        except PermissionError as e:
            logger.critical(
                f"[ai_gateway] embed_chunks: AUTH FAILURE chunk_id={chunk_id} file_name={file_name} — "
                f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
            )
            raise RuntimeError("AI Gateway authentication failed — aborting embedding.") from e

        except requests.exceptions.Timeout as e:
            reason = f"Timeout after {REQUEST_TIMEOUT}s"
            failure_reasons.append(reason)
            logger.error(
                f"[ai_gateway] embed_chunks: TIMEOUT chunk_id={chunk_id} file_name={file_name} — "
                f"error={e}\n{traceback.format_exc()}"
            )

        except requests.exceptions.ConnectionError as e:
            reason = f"ConnectionError: {e}"
            failure_reasons.append(reason)
            logger.error(
                f"[ai_gateway] embed_chunks: CONNECTION ERROR chunk_id={chunk_id} file_name={file_name} — "
                f"error={e}\n{traceback.format_exc()}"
            )

        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else "unknown"
            body   = e.response.text[:300] if e.response is not None else ""
            reason = f"HTTP {status}: {body[:120]}"
            failure_reasons.append(reason)
            logger.error(
                f"[ai_gateway] embed_chunks: HTTP ERROR chunk_id={chunk_id} file_name={file_name} "
                f"status={status} body={body} — {e}\n{traceback.format_exc()}"
            )

        except ValueError as e:
            reason = f"ValueError: {e}"
            failure_reasons.append(reason)
            logger.warning(
                f"[ai_gateway] embed_chunks: SKIPPED chunk_id={chunk_id} file_name={file_name} "
                f"reason={e}\n{traceback.format_exc()}"
            )

        except Exception as e:
            reason = f"{type(e).__name__}: {e}"
            failure_reasons.append(reason)
            logger.error(
                f"[ai_gateway] embed_chunks: UNEXPECTED ERROR chunk_id={chunk_id} file_name={file_name} "
                f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
            )

    failed = len(chunks) - len(results)
    logger.info(
        f"[ai_gateway] embed_chunks: done — {len(results)} OK, "
        f"{failed} failed out of {len(chunks)}"
    )

    # Surface distinct failure reasons so callers can log/raise them meaningfully
    if failure_reasons:
        # Collect the file names of failed chunks for easy CloudWatch debugging
        failed_files = list(dict.fromkeys(
            c.get("file_name", "unknown")
            for c in chunks
            if "embedding" not in c
        ))

        # Deduplicate while preserving order for the summary
        seen: dict = {}
        for r in failure_reasons:
            seen[r] = seen.get(r, 0) + 1
        summary = "; ".join(f"{r} (x{n})" if n > 1 else r for r, n in seen.items())
        logger.error(
            f"[ai_gateway] embed_chunks: failure summary — "
            f"failed_files={failed_files} reasons={summary}"
        )
        # Attach to the return value so callers can surface it without re-parsing logs
        _embed_chunks_last_failure_summary[0] = summary

    return results


# Mutable container so callers can inspect the last failure summary without
# requiring a return-value schema change.
_embed_chunks_last_failure_summary: List[str] = [""]


def _embed_single_chunk(chunk: Dict[str, Any]) -> List[float]:
    """
    Call the AI Gateway to embed a single chunk.

    Supports three modes depending on what the chunk carries:

    - Text-only  (image_path is None / missing):
        payload  → {"model": model, "input": "<text>"}

    - Image-only (text_content is empty, image_path is set):
        payload  → {"model": model,
                     "input": {"image": {"bytesBase64Encoded": "<b64>"}}}

    - Multimodal (both text and image_path are present):
        payload  → {"model": model,
                     "input": {"text": "<text>",
                                "image": {"bytesBase64Encoded": "<b64>"}}}

    The response shape is the same in all three cases:
        body["data"][0]["embedding"]
    """
    import base64

    chunk_id   = chunk.get("chunk_id", "unknown")
    text       = (chunk.get("text_content") or "").strip()
    image_path = chunk.get("image_path")

    # ── Input validation ──────────────────────────────────────────────────────
    has_text  = bool(text)
    has_image = bool(image_path)

    if not has_text and not has_image:
        raise ValueError(
            f"chunk_id={chunk_id} has neither text_content nor image_path — "
            "cannot embed an empty chunk."
        )

    # Safety guard for text: gemini-multimodalembedding rejects inputs > 1024 chars.
    # file_handler should have already split these, but truncate here as a last resort.
    _MAX_CHARS = 1000
    if has_text and len(text) > _MAX_CHARS:
        logger.warning(
            f"[ai_gateway] _embed_single_chunk: chunk_id={chunk_id} text too long "
            f"({len(text)} chars > {_MAX_CHARS}) — truncating before embedding."
        )
        text = text[:_MAX_CHARS].rsplit(" ", 1)[0]  # break at word boundary

    mode = "text+image" if (has_text and has_image) else ("image" if has_image else "text")
    logger.info(
        f"[ai_gateway] _embed_single_chunk: chunk_id={chunk_id} "
        f"mode={mode} text_len={len(text)} image={image_path}"
    )

    # ── Resolve config and token ──────────────────────────────────────────────
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

    # ── Build payload ─────────────────────────────────────────────────────────
    if mode == "text":
        # Text-only — preserve the original flat string format.
        payload = {"model": model, "input": text}

    else:
        # Image or multimodal — build the structured input dict.
        try:
            with open(image_path, "rb") as img_f:
                b64_image = base64.b64encode(img_f.read()).decode("utf-8")
        except FileNotFoundError:
            raise RuntimeError(
                f"Image file not found for chunk_id={chunk_id}: '{image_path}'. "
                "Ensure process_document() ran before embed_chunks()."
            )
        except OSError as e:
            raise RuntimeError(
                f"Could not read image file '{image_path}' for "
                f"chunk_id={chunk_id}: {e}"
            ) from e

        input_body: Dict[str, Any] = {
            "image": {"bytesBase64Encoded": b64_image}
        }
        if has_text:
            input_body["text"] = text

        payload = {"model": model, "input": input_body}

    logger.info(
        f"[ai_gateway] _embed_single_chunk: POST {url} model={model} "
        f"mode={mode} timeout={REQUEST_TIMEOUT}s"
    )

    # ── HTTP call ─────────────────────────────────────────────────────────────
    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json=payload,
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

    # ── Parse response ────────────────────────────────────────────────────────
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
        f"mode={mode} dim={len(embedding)} model={body.get('model')} "
        f"tokens={body.get('usage', {}).get('total_tokens')}"
    )
    return embedding


def embed_query(text: str) -> List[float]:

    if not text or not text.strip():
        raise ValueError("embed_query: text must be a non-empty string.")

    logger.info(f"[ai_gateway] embed_query: text_length={len(text.strip())}")

    # Reuse _embed_single_chunk — give it a stable sentinel chunk_id for logs
    chunk = {"chunk_id": "__query__", "text_content": text.strip()}
    try:
        vector = _embed_single_chunk(chunk)
    except Exception as e:
        logger.error(
            f"[ai_gateway] embed_query: failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise

    logger.info(f"[ai_gateway] embed_query: OK dim={len(vector)}")
    return vector


def generate_answer(question: str, context_chunks: List[Dict[str, Any]]) -> str:

    if not question or not question.strip():
        raise ValueError("generate_answer: question must be non-empty.")
    if not context_chunks:
        raise ValueError("generate_answer: context_chunks must not be empty.")

    logger.info(
        f"[ai_gateway] generate_answer: question_length={len(question)} "
        f"num_chunks={len(context_chunks)}"
    )

    # Resolve credentials + config
    try:
        api_key = _get_api_key()
        config  = get_config()
    except Exception as e:
        logger.error(
            f"[ai_gateway] generate_answer: config/key resolution failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise

    try:
        chat_model = config.get("chat_model", "gemini-pro")
        url        = f"{config['ai_gateway_url'].rstrip('/')}/chat/completions"
    except KeyError as e:
        raise RuntimeError(f"Config is missing required key: {e}") from e

    # Build the context block from retrieved chunks
    context_parts: List[str] = []
    for i, chunk in enumerate(context_chunks, start=1):
        fname   = chunk.get("file_name", "unknown document")
        page    = chunk.get("page_number")
        text    = chunk.get("text_content", "").strip()
        src_hdr = f"[Source {i}: {fname}" + (f", page {page}" if page else "") + "]"
        context_parts.append(f"{src_hdr}\n{text}")

    context_str = "\n\n".join(context_parts)

    system_prompt = (
        "You are a precise HR policy assistant for Titan Company.\n"
        "Answer the employee's question using ONLY the information in the HR policy "
        "documents provided below.\n\n"
        "Rules:\n"
        "- Lead with a direct, specific answer (e.g. exact number of days, "
        "eligibility criteria, effective date).\n"
        "- Quote or closely paraphrase the relevant policy text; include exact "
        "figures, dates, and conditions as written in the source.\n"
        "- If the question has multiple parts, address each part in turn.\n"
        "- If several source documents give relevant but different details, "
        "synthesise them and note any differences.\n"
        "- Do NOT infer, assume, or add anything not explicitly stated in the documents.\n"
        "- If the documents do not contain enough information to answer, state clearly "
        "what is missing and direct the employee to contact the HR department directly."
    )

    user_message = (
        f"Employee question: {question.strip()}\n\n"
        f"HR policy documents:\n{context_str}\n\n"
        f"Answer:"
    )

    logger.info(
        f"[ai_gateway] generate_answer: POST {url} model={chat_model} "
        f"context_chars={len(context_str)} timeout={REQUEST_TIMEOUT}s"
    )

    try:
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type":  "application/json",
            },
            json={
                "model": chat_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
                "temperature": 0.2,
                "max_tokens":  1024,
            },
            timeout=REQUEST_TIMEOUT,
        )
    except requests.exceptions.Timeout as e:
        logger.error(
            f"[ai_gateway] generate_answer: request timed out after "
            f"{REQUEST_TIMEOUT}s url={url}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"AI Gateway chat request timed out after {REQUEST_TIMEOUT}s.") from e
    except requests.exceptions.ConnectionError as e:
        logger.error(
            f"[ai_gateway] generate_answer: cannot connect to {url} — "
            f"error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Cannot connect to AI Gateway at {url}.") from e
    except Exception as e:
        logger.error(
            f"[ai_gateway] generate_answer: requests.post failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise

    logger.info(
        f"[ai_gateway] generate_answer: response status={response.status_code}"
    )

    if response.status_code in (401, 403):
        logger.error(
            f"[ai_gateway] generate_answer: auth rejected "
            f"status={response.status_code} body={response.text[:300]}"
        )
        raise PermissionError(
            f"AI Gateway returned {response.status_code} on chat/completions: "
            f"{response.text[:200]}"
        )

    if not response.ok:
        logger.error(
            f"[ai_gateway] generate_answer: non-2xx "
            f"status={response.status_code} body={response.text[:300]}"
        )
        response.raise_for_status()

    try:
        body   = response.json()
        answer = body["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as e:
        logger.error(
            f"[ai_gateway] generate_answer: unexpected response shape — "
            f"type={type(e).__name__} error={e} body={response.text[:300]}\n"
            f"{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"Unexpected AI Gateway chat response shape: {e}. "
            f"Body: {response.text[:200]}"
        ) from e

    logger.info(
        f"[ai_gateway] generate_answer: OK answer_length={len(answer)} "
        f"model={body.get('model')} "
        f"tokens={body.get('usage', {}).get('total_tokens')}"
    )
    return answer
