"""
PDF ingestion pipeline with duplicate detection.

Workflow:
1. Validate uploaded PDF.
2. Read file bytes.
3. Calculate SHA-256 hash.
4. Check document registry.
5. Reuse existing document if already processed.
6. Otherwise save and extract the PDF.
7. Register the document only after successful processing.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile

from app.document_registry import (
    calculate_file_hash,
    get_document_by_hash,
    register_document,
)
from app.ingestion.metadata import DocumentMetadata
from app.ingestion.pdf_extractor import extract_pdf_text
from app.ingestion.pdf_validator import validate_pdf_file


PROJECT_ROOT = Path(__file__).resolve().parents[3]

UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "uploads"
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"

UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)


def _processed_file_exists(document: dict[str, Any]) -> bool:
    """
    Check whether the processed JSON file still exists.
    """

    processed_file = document.get("processed_file")

    if not processed_file:
        return False

    return Path(processed_file).exists()


async def process_uploaded_pdf(
    file: UploadFile,
) -> dict[str, Any]:
    """
    Validate, process, and register an uploaded PDF.

    Duplicate behavior:
    If the same file hash already exists and its processed JSON
    file is available, the existing document is reused.
    """

    # ---------------------------------------------------------
    # 1. Validate file type and basic upload requirements
    # ---------------------------------------------------------
    validate_pdf_file(file)

    # ---------------------------------------------------------
    # 2. Read uploaded file content
    # ---------------------------------------------------------
    file_content = await file.read()

    if not file_content:
        raise ValueError("Uploaded PDF is empty.")

    # ---------------------------------------------------------
    # 3. Calculate deterministic file hash
    # ---------------------------------------------------------
    file_hash = calculate_file_hash(file_content)

    # ---------------------------------------------------------
    # 4. Check whether this file was already processed
    # ---------------------------------------------------------
    existing_document = get_document_by_hash(file_hash)

    if existing_document is not None:
        if _processed_file_exists(existing_document):
            return {
                "status": "reused",
                "message": (
                    "This PDF was already processed. "
                    "The existing document has been reused."
                ),
                "document": existing_document,
                "extraction_summary": {
                    "page_count": existing_document.get("page_count"),
                    "total_characters": existing_document.get(
                        "total_characters"
                    ),
                },
                "processed_file": existing_document.get(
                    "processed_file"
                ),
                "duplicate": True,
            }

    # ---------------------------------------------------------
    # 5. Create a new document ID
    # ---------------------------------------------------------
    document_id = uuid4()
    uploaded_at = datetime.now(timezone.utc)

    safe_filename = Path(
        file.filename or "uploaded_document.pdf"
    ).name

    saved_pdf_path = (
        UPLOAD_DIRECTORY
        / f"{document_id}_{safe_filename}"
    )

    # ---------------------------------------------------------
    # 6. Save original PDF
    # ---------------------------------------------------------
    saved_pdf_path.write_bytes(file_content)

    # ---------------------------------------------------------
    # 7. Extract text from PDF
    # ---------------------------------------------------------
    extraction_result = extract_pdf_text(saved_pdf_path)

    # ---------------------------------------------------------
    # 8. Create document metadata
    # ---------------------------------------------------------
    metadata = DocumentMetadata(
        document_id=str(document_id),
        filename=safe_filename,
        file_size_bytes=len(file_content),
        content_type=file.content_type or "application/pdf",
        page_count=extraction_result["page_count"],
        total_characters=extraction_result[
            "total_characters"
        ],
        uploaded_at=uploaded_at,
        ingestion_status="text_extracted",
        source_type="uploaded_pdf",
        extraction_method=extraction_result[
            "extraction_method"
        ],
        metadata_version="1.0",
    )

    metadata_dict = metadata.model_dump(mode="json")

    # ---------------------------------------------------------
    # 9. Persist processed document JSON
    # ---------------------------------------------------------
    processed_document = {
        "document": metadata_dict,
        "extraction": extraction_result,
        "processed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    processed_json_path = (
        PROCESSED_DIRECTORY
        / f"{document_id}.json"
    )

    processed_json_path.write_text(
        json.dumps(
            processed_document,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # 10. Register only after successful processing
    # ---------------------------------------------------------
    registry_record = register_document(
        file_hash=file_hash,
        document_data={
            **metadata_dict,
            "status": "ready",
            "processed_file": str(
                processed_json_path
            ),
            "uploaded_pdf": str(
                saved_pdf_path
            ),
            "page_count": extraction_result[
                "page_count"
            ],
            "total_characters": extraction_result[
                "total_characters"
            ],
        },
    )

    return {
        "status": "success",
        "message": (
            "PDF uploaded, text extracted, "
            "and processed data saved successfully."
        ),
        "document": registry_record,
        "extraction_summary": {
            "page_count": extraction_result[
                "page_count"
            ],
            "total_characters": extraction_result[
                "total_characters"
            ],
        },
        "processed_file": str(processed_json_path),
        "duplicate": False,
    }