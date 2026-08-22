from unittest.mock import patch

import pytest

from app.document_loader import DocumentExtractionError, extract_pdf_document, source_id_from_filename


def test_source_id_is_filename_safe() -> None:
    assert source_id_from_filename("Phil Spencer email (final).pdf") == "phil-spencer-email-final"


def test_pdf_extraction_builds_ingest_document() -> None:
    fake_reader = type(
        "Reader",
        (),
        {"pages": [type("Page", (), {"extract_text": lambda self: "Microsoft acquired Activision."})()]},
    )
    with patch("app.document_loader.PdfReader", return_value=fake_reader):
        document = extract_pdf_document("acquisition.pdf", b"%PDF example")

    assert document.source_id == "acquisition"
    assert document.content == "Microsoft acquired Activision."
    assert document.metadata["pages"] == "1"


def test_pdf_without_selectable_text_is_rejected() -> None:
    fake_reader = type("Reader", (), {"pages": [type("Page", (), {"extract_text": lambda self: ""})()]})
    with patch("app.document_loader.PdfReader", return_value=fake_reader):
        with pytest.raises(DocumentExtractionError, match="No selectable text"):
            extract_pdf_document("scanned.pdf", b"%PDF example")
