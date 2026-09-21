"""
PDF upload validation utilities.

This module contains synchronous validation logic for uploaded PDF files.
"""

from pathlib import Path

from fastapi import HTTPException, UploadFile


# Supported MIME types.
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
}

# Maximum allowed upload size: 20 MB.
MAX_FILE_SIZE_BYTES = 20 * 1024 * 1024


def validate_pdf_file(file: UploadFile) -> None:
    """
    Validate an uploaded PDF file.

    Validation checks:
    1. A file was provided.
    2. The file has a valid filename.
    3. The filename ends with .pdf.
    4. The content type is application/pdf.
    5. The file size does not exceed the configured limit.

    This function is intentionally synchronous.
    """

    if file is None:
        raise HTTPException(
            status_code=400,
            detail="No file was uploaded.",
        )

    filename = Path(file.filename or "").name

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must have a filename.",
        )

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid content type. "
                "The uploaded file must have application/pdf content type."
            ),
        )

    # Check the size when the UploadFile object exposes a file stream.
    file_object = getattr(file, "file", None)

    if file_object is not None:
        current_position = file_object.tell()

        file_object.seek(0, 2)
        file_size = file_object.tell()

        file_object.seek(current_position)

        if file_size > MAX_FILE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail="PDF file size must not exceed 20 MB.",
            )