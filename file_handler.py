import os
import re
import uuid
import logging
import traceback
import unicodedata
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Chunking config:
# 180 words × ~5 chars/word ≈ 900 chars — safely under the 1000-char embedding
# limit for most chunks, so _split_long_text rarely triggers.
CHUNK_TOKEN_SIZE = 180    # target words per chunk
CHUNK_OVERLAP    = 40     # overlap words between consecutive chunks

# Hard character limit imposed by the embedding model (Vertex AI gemini-multimodalembedding).
MAX_EMBED_CHARS = 1000


def process_document(local_path: str, metadata: dict) -> List[Dict[str, Any]]:

    logger.info(
        f"[file_handler] process_document: local_path='{local_path}' "
        f"metadata_keys={list(metadata.keys()) if metadata else None}"
    )

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




def _preprocess_text(text: str) -> str:

    if not text:
        return ""

    # 1. Unicode normalisation — resolves ligatures (ﬁ→fi), full-width chars, etc.
    text = unicodedata.normalize("NFKC", text)

    # 2. Standalone page numbers: "3", "Page 3", "Page 3 of 12" on their own line
    text = re.sub(
        r"(?im)^\s*(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?\s*$",
        "",
        text,
    )

    # 3. Common document boilerplate
    text = re.sub(
        r"(?i)(confidential|internal[\s-]+use[\s-]+only|for[\s-]+internal[\s-]+distribution"
        r"|titan[\s-]+company[\s-]+limited|all[\s-]+rights[\s-]+reserved|©\s*\d{4})",
        "",
        text,
    )

    # 4. Control characters (ASCII 0-8, 11-12, 14-31, 127) — preserve \t (9) and \n (10)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 5. Bullet and arrow symbols → space
    text = re.sub(r"[•▪▸►▶→‣◦◉●■□▫–—]", " ", text)

    # 6. Whitespace normalisation
    text = re.sub(r"[ \t]+", " ", text)        # multiple spaces/tabs → single space
    text = re.sub(r"\n{3,}", "\n\n", text)      # cap consecutive newlines at 2
    text = re.sub(r" *\n *", "\n", text)        # trim spaces around newlines

    return text.strip()


def _process_pdf(local_path: str, metadata: dict) -> List[Dict[str, Any]]:

    logger.info(f"[file_handler] _process_pdf: starting for '{local_path}'")

    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as e:
        raise ImportError(
            "pypdf is required for PDF processing. "
            "Install it with: pip install pypdf"
        ) from e

    try:
        reader = PdfReader(local_path)
    except PdfReadError as e:
        raise RuntimeError(
            f"PDF is corrupt or unreadable: {local_path}. Error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF {local_path}: {e}") from e

    if reader.is_encrypted:
        raise ValueError(f"PDF is encrypted and cannot be processed: {local_path}")

    page_count = len(reader.pages)
    if page_count == 0:
        raise ValueError(f"PDF has no pages: {local_path}")

    logger.info(f"[file_handler] _process_pdf: PDF opened — {page_count} page(s)")

    # ── Collect (word, page_number) tuples across all pages ──────────────────
    # This allows the chunker to slide across page boundaries, giving each
    # chunk full sentences instead of page-truncated fragments.
    page_words: List[Tuple[str, int]] = []

    for page_num, page in enumerate(reader.pages):
        try:
            raw_text = (page.extract_text() or "").strip()
        except Exception as e:
            logger.warning(
                f"[file_handler] _process_pdf: text extraction failed for "
                f"page {page_num + 1} — {type(e).__name__}: {e}. Skipping page."
            )
            raw_text = ""

        clean = _preprocess_text(raw_text)
        for word in clean.split():
            page_words.append((word, page_num + 1))

    if not page_words:
        # ── Scanned / image-only PDF ─────────────────────────────────────────
        # pypdf found no text layer.  Fall back to rendering each page as a PNG
        # so the multimodal embedding model (gemini-multimodalembedding) can
        # embed the visual content directly.
        logger.warning(
            f"[file_handler] _process_pdf: no text extracted from '{local_path}' — "
            f"falling back to per-page image rendering for multimodal embedding."
        )
        return _process_pdf_as_images(local_path, metadata, page_count)

    logger.info(
        f"[file_handler] _process_pdf: {len(page_words)} word(s) collected "
        f"across {page_count} page(s)"
    )

    chunks: List[Dict[str, Any]] = []
    start = 0

    while start < len(page_words):
        end    = min(start + CHUNK_TOKEN_SIZE, len(page_words))
        window = page_words[start:end]

        chunk_text = " ".join(w for w, _ in window)
        chunk_page = window[0][1]   # page number of the first word in this chunk

        # Guard: split any window that somehow still exceeds the char limit
        sub_texts = _split_long_text(chunk_text) if chunk_text else []
        for sub_idx, sub_text in enumerate(sub_texts):
            chunks.append({
                **metadata,
                "chunk_id":     str(uuid.uuid4()),
                "page_number":  chunk_page,
                "chunk_index":  len(chunks),
                "sub_chunk":    sub_idx if len(sub_texts) > 1 else None,
                "text_content": sub_text,
                "image_path":   None,
            })

        if end == len(page_words):
            break
        start = end - CHUNK_OVERLAP

    if not chunks:
        raise RuntimeError(f"PDF chunking produced no output for: {local_path}")

    logger.info(
        f"[file_handler] _process_pdf: SUCCESS — {len(chunks)} chunk(s) "
        f"from {page_count} page(s)"
    )
    return chunks




def _process_docx(local_path: str, metadata: dict) -> List[Dict[str, Any]]:

    logger.info(f"[file_handler] _process_docx: starting for '{local_path}'")

    try:
        from docx import Document
        from docx.opc.exceptions import PackageNotFoundError
    except ImportError as e:
        raise ImportError(
            "python-docx is required for DOCX processing. "
            "Install it with: pip install python-docx"
        ) from e

    try:
        doc = Document(local_path)
    except PackageNotFoundError as e:
        raise RuntimeError(
            f"DOCX file is corrupt or not a valid Office Open XML document: "
            f"{local_path}. Error: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to open DOCX {local_path}: {e}") from e

    text_blocks: List[str] = []

    try:
        for p in doc.paragraphs:
            clean = _preprocess_text(p.text)
            if clean:
                text_blocks.append(clean)
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract paragraphs from {local_path}: {e}"
        ) from e

    try:
        for table in doc.tables:
            for row in table.rows:
                cells = [
                    _preprocess_text(cell.text)
                    for cell in row.cells
                    if cell.text.strip()
                ]
                row_text = " | ".join(c for c in cells if c)
                if row_text.strip():
                    text_blocks.append(row_text)
    except Exception as e:
        logger.warning(
            f"[file_handler] _process_docx: table extraction failed — "
            f"{type(e).__name__}: {e}. Continuing without table data."
        )

    if not text_blocks:
        raise ValueError(
            f"DOCX contains no extractable text: {local_path}. "
            "The document may be empty or contain only images."
        )

    logger.info(
        f"[file_handler] _process_docx: {len(text_blocks)} text block(s) extracted "
        f"(paragraphs + table rows)"
    )

    try:
        raw_chunks = _semantic_chunk(text_blocks)
    except Exception as e:
        raise RuntimeError(f"Semantic chunking failed for {local_path}: {e}") from e

    if not raw_chunks:
        raise ValueError(f"Chunking produced no output for {local_path}.")

    chunks: List[Dict[str, Any]] = []
    for text in raw_chunks:
        sub_texts = _split_long_text(text)
        for sub_idx, sub_text in enumerate(sub_texts):
            chunks.append({
                **metadata,
                "chunk_id":     str(uuid.uuid4()),
                "page_number":  None,   # not meaningful for DOCX
                "chunk_index":  len(chunks),
                "sub_chunk":    sub_idx if len(sub_texts) > 1 else None,
                "text_content": sub_text,
                "image_path":   None,
            })

    logger.info(
        f"[file_handler] _process_docx: SUCCESS — {len(chunks)} chunk(s) created"
    )
    return chunks


# ---------------------------------------------------------------------------
# Image fallback — scanned / image-only PDFs
# ---------------------------------------------------------------------------

def _process_pdf_as_images(local_path: str, metadata: dict, page_count: int) -> List[Dict[str, Any]]:
    """
    Render each page of a scanned (text-less) PDF to a PNG and return one
    chunk per page with ``image_path`` set.  ``text_content`` is left as an
    empty string so the multimodal embedding model receives the image directly.

    Images are written to /tmp/ using a UUID prefix so concurrent Lambda
    invocations never collide, and s3_utils.cleanup_tmp() already purges
    page_*.png files from /tmp/.

    Raises:
        ImportError:   pymupdf not installed.
        RuntimeError:  Rendering fails or produces no pages.
    """
    logger.info(
        f"[file_handler] _process_pdf_as_images: rendering {page_count} page(s) "
        f"of '{local_path}' to PNG images"
    )

    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise ImportError(
            "pymupdf is required to process scanned PDFs. "
            "Install it with: pip install pymupdf"
        ) from e

    try:
        doc = fitz.open(local_path)
    except Exception as e:
        raise RuntimeError(f"pymupdf could not open '{local_path}': {e}") from e

    # Use a short document-scoped UUID so all page images for this doc share
    # a common prefix — makes cleanup easy and avoids cross-doc name collisions.
    doc_uuid = str(uuid.uuid4())[:8]
    # 150 DPI: good balance between image quality and payload size.
    # gemini-multimodalembedding handles images up to ~20 MB; a 150 DPI A4
    # page PNG is typically 200–500 KB.
    RENDER_DPI = 150
    zoom       = RENDER_DPI / 72.0
    matrix     = fitz.Matrix(zoom, zoom)

    chunks: List[Dict[str, Any]] = []

    try:
        for page_num in range(len(doc)):
            page     = doc[page_num]
            pix      = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            img_path = os.path.join("/tmp", f"page_{doc_uuid}_{page_num}.png")

            try:
                pix.save(img_path)
            except Exception as e:
                logger.warning(
                    f"[file_handler] _process_pdf_as_images: could not save page "
                    f"{page_num + 1} image to '{img_path}' — {type(e).__name__}: {e}. Skipping."
                )
                continue

            logger.debug(
                f"[file_handler] _process_pdf_as_images: page {page_num + 1} → "
                f"{img_path} ({pix.width}×{pix.height} px)"
            )

            chunks.append({
                **metadata,
                "chunk_id":     str(uuid.uuid4()),
                "page_number":  page_num + 1,
                "chunk_index":  page_num,
                "sub_chunk":    None,
                "text_content": "",        # no text — embedding is image-only
                "image_path":   img_path,
            })
    finally:
        doc.close()

    if not chunks:
        raise RuntimeError(
            f"PDF image rendering produced no pages for '{local_path}'. "
            "The file may be corrupt or have zero renderable pages."
        )

    logger.info(
        f"[file_handler] _process_pdf_as_images: SUCCESS — "
        f"{len(chunks)} image chunk(s) from {page_count} page(s)"
    )
    return chunks


# ---------------------------------------------------------------------------
# Shared chunking helpers
# ---------------------------------------------------------------------------

def _split_long_text(text: str, max_chars: int = MAX_EMBED_CHARS) -> List[str]:
    """
    Split *text* into pieces no longer than *max_chars*, breaking at word
    boundaries where possible.  Returns a list with at least one element.

    This is a safety net — with CHUNK_TOKEN_SIZE=180 most chunks are ~900 chars
    and will pass through unchanged.
    """
    if len(text) <= max_chars:
        return [text]

    pieces: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            pieces.append(remaining)
            break
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at <= 0:
            split_at = max_chars   # no space found — hard cut
        pieces.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()

    return [p for p in pieces if p]


def _semantic_chunk(paragraphs: List[str]) -> List[str]:

    logger.info(
        f"[file_handler] _semantic_chunk: chunking {len(paragraphs)} block(s) "
        f"(size={CHUNK_TOKEN_SIZE}, overlap={CHUNK_OVERLAP})"
    )

    try:
        if not isinstance(paragraphs, list):
            raise ValueError(
                f"_semantic_chunk expected a list, got {type(paragraphs).__name__}."
            )

        if CHUNK_TOKEN_SIZE <= 0:
            raise ValueError(f"CHUNK_TOKEN_SIZE must be > 0, got {CHUNK_TOKEN_SIZE}.")
        if CHUNK_OVERLAP < 0:
            raise ValueError(f"CHUNK_OVERLAP must be >= 0, got {CHUNK_OVERLAP}.")
        if CHUNK_OVERLAP >= CHUNK_TOKEN_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than "
                f"CHUNK_TOKEN_SIZE ({CHUNK_TOKEN_SIZE})."
            )

        all_words: List[str] = []
        for para in paragraphs:
            if isinstance(para, str):
                all_words.extend(para.split())
            else:
                logger.warning(
                    f"[file_handler] _semantic_chunk: skipping non-string block: {para!r}"
                )

        if not all_words:
            logger.warning("[file_handler] _semantic_chunk: no words — returning []")
            return []

        chunks: List[str] = []
        start = 0

        while start < len(all_words):
            end = min(start + CHUNK_TOKEN_SIZE, len(all_words))
            chunks.append(" ".join(all_words[start:end]))

            if end == len(all_words):
                break

            start = end - CHUNK_OVERLAP

        logger.info(
            f"[file_handler] _semantic_chunk: {len(chunks)} chunk(s) "
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
