"""
Tests for duplicate PDF detection.

These tests verify that:
1. An already-registered PDF is detected using its file hash.
2. The existing processed document is reused.
3. PDF extraction is not executed again for duplicate files.
4. The duplicate response contains the expected metadata.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from app.ingestion.pipeline import process_uploaded_pdf


class FakeUploadFile:
    """
    Minimal fake UploadFile implementation for testing.

    The production pipeline only requires:
    - filename
    - content_type
    - async read()
    """

    def __init__(
        self,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
    ):
        self.filename = filename
        self.content_type = content_type
        self._content = content

    async def read(self) -> bytes:
        """Return the uploaded file content."""
        return self._content


@pytest.mark.asyncio
async def test_duplicate_pdf_is_reused():
    """
    Verify that an already-processed PDF is reused.

    Expected behavior:
    - Existing document is found by hash.
    - Existing processed JSON file exists.
    - PDF extraction is not called again.
    - Pipeline returns status='reused'.
    - duplicate=True is returned.
    """

    pdf_content = b"%PDF-1.4 fake PDF content for duplicate testing"

    existing_processed_file = Path(__file__).resolve().parent / (
        "fake_existing_processed_document.json"
    )

    existing_document = {
        "document_id": "existing-document-123",
        "filename": "existing_document.pdf",
        "status": "ready",
        "page_count": 3,
        "total_characters": 2500,
        "processed_file": str(existing_processed_file),
    }

    uploaded_file = FakeUploadFile(
        filename="duplicate_document.pdf",
        content=pdf_content,
    )

    with (
        patch(
            "app.ingestion.pipeline.validate_pdf_file",
            return_value=None,
        ) as mock_validate_pdf,
        patch(
            "app.ingestion.pipeline.get_document_by_hash",
            return_value=existing_document,
        ) as mock_get_document,
        patch(
            "app.ingestion.pipeline.extract_pdf_text",
        ) as mock_extract_pdf,
    ):
        # Simulate that the existing processed JSON file is available.
        with patch(
            "app.ingestion.pipeline._processed_file_exists",
            return_value=True,
        ):
            result = await process_uploaded_pdf(uploaded_file)

    # Validation must be called once.
    mock_validate_pdf.assert_called_once_with(uploaded_file)

    # Registry lookup must be performed.
    mock_get_document.assert_called_once()

    # Extraction must not run again for a duplicate document.
    mock_extract_pdf.assert_not_called()

    # Validate the response.
    assert result["status"] == "reused"
    assert result["duplicate"] is True
    assert result["document"] == existing_document
    assert result["processed_file"] == existing_document["processed_file"]

    assert result["extraction_summary"]["page_count"] == 3
    assert result["extraction_summary"]["total_characters"] == 2500

    assert "already processed" in result["message"]


@pytest.mark.asyncio
async def test_new_pdf_is_not_marked_as_duplicate():
    """
    Verify that a PDF not found in the registry is processed as new.

    This test mocks the extraction step so that no real PDF file is
    required.
    """

    pdf_content = b"%PDF-1.4 fake new PDF content"

    extraction_result = {
        "page_count": 2,
        "total_characters": 1200,
        "extraction_method": "pypdf",
        "pages": [
            {
                "page_number": 1,
                "text": "First page content",
                "character_count": 900,
            },
            {
                "page_number": 2,
                "text": "Second page content",
                "character_count": 300,
            },
        ],
    }

    registered_document = {
        "document_id": "new-document-456",
        "filename": "new_document.pdf",
        "status": "ready",
        "page_count": 2,
        "total_characters": 1200,
    }

    uploaded_file = FakeUploadFile(
        filename="new_document.pdf",
        content=pdf_content,
    )

    with (
        patch(
            "app.ingestion.pipeline.validate_pdf_file",
            return_value=None,
        ) as mock_validate_pdf,
        patch(
            "app.ingestion.pipeline.get_document_by_hash",
            return_value=None,
        ) as mock_get_document,
        patch(
            "app.ingestion.pipeline.extract_pdf_text",
            return_value=extraction_result,
        ) as mock_extract_pdf,
        patch(
            "app.ingestion.pipeline.register_document",
            return_value=registered_document,
        ) as mock_register_document,
    ):
        result = await process_uploaded_pdf(uploaded_file)

    # Validation and registry lookup should occur.
    mock_validate_pdf.assert_called_once_with(uploaded_file)
    mock_get_document.assert_called_once()

    # New documents must be extracted.
    mock_extract_pdf.assert_called_once()

    # New documents must be registered.
    mock_register_document.assert_called_once()

    # Validate the response.
    assert result["status"] == "success"
    assert result["duplicate"] is False
    assert result["document"] == registered_document

    assert result["extraction_summary"]["page_count"] == 2
    assert result["extraction_summary"]["total_characters"] == 1200