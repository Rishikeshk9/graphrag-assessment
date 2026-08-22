"""Safe, text-first document extraction for ingestion endpoints."""

from __future__ import annotations

import hashlib
import io
import re

from pypdf import PdfReader

from app.schemas import DocumentInput

MAX_PDF_BYTES = 20 * 1024 * 1024
SOURCE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_-]+")


class DocumentExtractionError(ValueError):
    """Raised when a document cannot safely yield text for RAG ingestion."""


def source_id_from_filename(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0].strip().lower()
    source_id = SOURCE_ID_PATTERN.sub("-", stem).strip("-")
    return source_id[:128] or "uploaded-document"


def file_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def extract_pdf_document(filename: str, payload: bytes) -> DocumentInput:
    if len(payload) > MAX_PDF_BYTES:
        raise DocumentExtractionError("PDF exceeds the 20 MB upload limit")
    if not payload.startswith(b"%PDF"):
        raise DocumentExtractionError("Uploaded file is not a valid PDF")
    try:
        reader = PdfReader(io.BytesIO(payload))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as error:
        raise DocumentExtractionError("Unable to read PDF content") from error

    content = "\n\n".join(page.strip() for page in pages if page.strip()).strip()
    if not content:
        raise DocumentExtractionError(
            "No selectable text found. Use a text-based PDF or add OCR support."
        )
    return DocumentInput(
        source_id=source_id_from_filename(filename),
        content=content,
        file_sha256=file_sha256(payload),
        metadata={"filename": filename, "file_type": "application/pdf", "pages": str(len(pages))},
    )
