import os
import uuid
import logging
import traceback
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# DOCX chunking config
CHUNK_TOKEN_SIZE = 500    # target tokens per chunk (word count proxy)
CHUNK_OVERLAP    = 200    # overlap tokens between consecutive chunks


def process_document(local_path: str, metadata: dict) -> List[Dict[str, Any]]:
    """
    Entry point: detect file type and route to the correct parser.

    Args:
        local_path: Absolute path to the downloaded file in /tmp.
        metadata:   Metadata dict from s3_utils (document_id, file_type, tags, etc.).

    Returns:
        List of chunk dicts.

    Raises:
        ValueError:    Unsupported file type or invalid arguments.
        FileNotFoundError: local_path does not exist.
        RuntimeError:  Parsing failed for an unexpected reason.
    """
    logger.info(f"[file_handler] process_document: local_path='{local_path}' metadata_keys={list(metadata.keys()) if metadata else None}")

    # --- Input validation ---
    try:
        if not local_path or not isinstance(local_path, str):
            raise ValueError(f"Invalid local_path: {local_path!r}")
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata must be a dict, got {type(metadata).__name__}.")

        if not os.path.exists(local_path):
            raise FileNotFoundError(
                f"Document not found at {local_path}. "
                "Ensure download_document() completed successfully."
            )
        if os.path.getsize(local_path) == 0:
            raise ValueError(f"Document at {local_path} is empty (0 bytes) — cannot parse.")

        file_type = metadata.get("file_type", "").lower().strip()
        if not file_type:
            raise ValueError(
                "metadata is missing 'file_type'. "
                "Ensure get_object_tags() ran before process_document()."
            )
    except (ValueError, FileNotFoundError):
        raise
    except Exception as e:
        logger.error(
            f"[file_handler] process_document: input validation failed — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Input validation error in process_document: {e}") from e

    logger.info(f"[file_handler] process_document: processing '{local_path}' (type={file_type})")

    try:
        if file_type == "pdf":
            return _process_pdf(local_path, metadata)
        elif file_type in ("docx", "doc"):
            return _process_docx(local_path, metadata)
        else:
            raise ValueError(
                f"Unsupported file type: '{file_type}'. "
                f"Supported types: pdf, docx, doc."
            )
    except (ValueError, ImportError, RuntimeError):
        raise
    except Exception as e:
        logger.error(
            f"[file_handler] process_document: unexpected error — "
            f"type={type(e).__name__} file_type='{file_type}' "
            f"local_path='{local_path}' error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to process document '{local_path}': {e}") from e


# ---------------------------------------------------------------------------
# PDF Processing  (pure-Python pypdf — no native binary dependencies)
# ---------------------------------------------------------------------------

def _process_pdf(local_path: str, metadata: dict) -> List[Dict[str, Any]]:
    """
    Extract text from each PDF page using pypdf.
    Returns one chunk per page.
    """
    logger.info(f"[file_handler] _process_pdf: starting for '{local_path}'")

    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as e:
        logger.error(
            f"[file_handler] _process_pdf: pypdf not installed — "
            f"error={e}\n{traceback.format_exc()}"
        )
        raise ImportError(
            "pypdf is required for PDF processing. "
            "Install it with: pip install pypdf"
        )

    # --- Open document ---
    try:
        reader = PdfReader(local_path)
    except PdfReadError as e:
        logger.error(
            f"[file_handler] _process_pdf: PDF corrupt or unreadable — "
            f"path='{local_path}' error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"PDF is corrupt or unreadable: {local_path}. Error: {e}"
        ) from e
    except Exception as e:
        logger.error(
            f"[file_handler] _process_pdf: failed to open PDF — "
            f"type={type(e).__name__} path='{local_path}' error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to open PDF {local_path}: {e}") from e

    # --- Encryption check ---
    try:
        if reader.is_encrypted:
            raise ValueError(
                f"PDF is encrypted and cannot be processed: {local_path}"
            )

        page_count = len(reader.pages)
        if page_count == 0:
            raise ValueError(f"PDF has no pages: {local_path}")
    except ValueError:
        raise
    except Exception as e:
        logger.error(
            f"[file_handler] _process_pdf: error checking PDF properties — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to inspect PDF {local_path}: {e}") from e

    logger.info(f"[file_handler] _process_pdf: PDF opened — {page_count} page(s)")

    chunks = []

    for page_num, page in enumerate(reader.pages):
        try:
            # Extract text (best-effort)
            try:
                text_content = (page.extract_text() or "").strip()
            except Exception as e:
                logger.warning(
                    f"[file_handler] _process_pdf: text extraction failed for "
                    f"page {page_num + 1} of '{local_path}' — "
                    f"type={type(e).__name__} error={e}. Continuing with empty text."
                )
                text_content = ""

            chunk = {
                **metadata,
                "chunk_id":     str(uuid.uuid4()),
                "page_number":  page_num + 1,   # 1-indexed
                "chunk_index":  page_num,
                "text_content": text_content,
                "image_path":   None,           # image rendering not used
            }
            chunks.append(chunk)

        except Exception as e:
            logger.error(
                f"[file_handler] _process_pdf: failed to process page {page_num + 1} — "
                f"type={type(e).__name__} error={e}\n{traceback.format_exc()}. Skipping page."
            )

    if not chunks:
        raise RuntimeError(
            f"PDF parsing produced no chunks for {local_path}. "
            "All pages may have failed to process."
        )

    logger.info(f"[file_handler] _process_pdf: SUCCESS — {len(chunks)}/{page_count} page chunk(s) created.")
    return chunks


# ---------------------------------------------------------------------------
# DOCX Processing
# ---------------------------------------------------------------------------

def _process_docx(local_path: str, metadata: dict) -> List[Dict[str, Any]]:
    """
    Extract paragraphs from a DOCX and produce semantic chunks.
    """
    logger.info(f"[file_handler] _process_docx: starting for '{local_path}'")

    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError as e:
        logger.error(
            f"[file_handler] _process_docx: python-docx not installed — "
            f"error={e}\n{traceback.format_exc()}"
        )
        raise ImportError(
            "python-docx is required for DOCX processing. "
            "Install it with: pip install python-docx"
        )

    # --- Open document ---
    try:
        doc = Document(local_path)
    except PackageNotFoundError as e:
        logger.error(
            f"[file_handler] _process_docx: DOCX corrupt or invalid — "
            f"path='{local_path}' error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"DOCX file is corrupt or not a valid Office Open XML document: "
            f"{local_path}. Error: {e}"
        ) from e
    except Exception as e:
        logger.error(
            f"[file_handler] _process_docx: failed to open DOCX — "
            f"type={type(e).__name__} path='{local_path}' error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to open DOCX {local_path}: {e}") from e

    # --- Extract paragraphs ---
    try:
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except Exception as e:
        logger.error(
            f"[file_handler] _process_docx: failed to extract paragraphs — "
            f"type={type(e).__name__} path='{local_path}' error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"Failed to extract paragraphs from {local_path}: {e}"
        ) from e

    if not paragraphs:
        raise ValueError(
            f"DOCX contains no extractable text paragraphs: {local_path}. "
            "The document may be empty or consist only of images/tables."
        )

    logger.info(
        f"[file_handler] _process_docx: {len(paragraphs)} non-empty paragraph(s) extracted"
    )

    # --- Chunk ---
    try:
        raw_chunks = _semantic_chunk(paragraphs)
    except Exception as e:
        logger.error(
            f"[file_handler] _process_docx: semantic chunking failed — "
            f"type={type(e).__name__} path='{local_path}' error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(
            f"Semantic chunking failed for {local_path}: {e}"
        ) from e

    if not raw_chunks:
        raise ValueError(f"Chunking produced no output for {local_path}.")

    try:
        chunks = []
        for idx, text in enumerate(raw_chunks):
            chunk = {
                **metadata,
                "chunk_id":     str(uuid.uuid4()),
                "page_number":  None,   # not applicable for DOCX
                "chunk_index":  idx,
                "text_content": text,
                "image_path":   None,
            }
            chunks.append(chunk)
    except Exception as e:
        logger.error(
            f"[file_handler] _process_docx: failed to build chunk dicts — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Failed to build DOCX chunks: {e}") from e

    logger.info(f"[file_handler] _process_docx: SUCCESS — {len(chunks)} semantic chunk(s) created.")
    return chunks


def _semantic_chunk(paragraphs: List[str]) -> List[str]:
    """
    Combine paragraphs into ~CHUNK_TOKEN_SIZE-word chunks with CHUNK_OVERLAP overlap.
    """
    logger.info(
        f"[file_handler] _semantic_chunk: chunking {len(paragraphs) if paragraphs else 0} paragraphs "
        f"(size={CHUNK_TOKEN_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    try:
        if not isinstance(paragraphs, list):
            raise ValueError(
                f"_semantic_chunk expected a list, got {type(paragraphs).__name__}."
            )

        all_words: List[str] = []
        for para in paragraphs:
            if isinstance(para, str):
                all_words.extend(para.split())
            else:
                logger.warning(f"[file_handler] _semantic_chunk: skipping non-string paragraph: {para!r}")

        if not all_words:
            logger.warning("[file_handler] _semantic_chunk: no words found — returning empty list")
            return []

        if CHUNK_TOKEN_SIZE <= 0:
            raise ValueError(f"CHUNK_TOKEN_SIZE must be > 0, got {CHUNK_TOKEN_SIZE}.")
        if CHUNK_OVERLAP < 0:
            raise ValueError(f"CHUNK_OVERLAP must be >= 0, got {CHUNK_OVERLAP}.")
        if CHUNK_OVERLAP >= CHUNK_TOKEN_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than "
                f"CHUNK_TOKEN_SIZE ({CHUNK_TOKEN_SIZE}) to avoid infinite loops."
            )

        chunks = []
        start  = 0

        while start < len(all_words):
            end         = min(start + CHUNK_TOKEN_SIZE, len(all_words))
            chunk_words = all_words[start:end]
            chunks.append(" ".join(chunk_words))

            if end == len(all_words):
                break

            start = end - CHUNK_OVERLAP

        logger.info(
            f"[file_handler] _semantic_chunk: produced {len(chunks)} chunk(s) "
            f"from {len(all_words)} words"
        )
        return chunks

    except ValueError:
        raise
    except Exception as e:
        logger.error(
            f"[file_handler] _semantic_chunk: unexpected error — "
            f"type={type(e).__name__} error={e}\n{traceback.format_exc()}"
        )
        raise RuntimeError(f"Semantic chunking failed unexpectedly: {e}") from e
