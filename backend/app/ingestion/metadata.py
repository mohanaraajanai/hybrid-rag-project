from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """
    Metadata associated with one uploaded PDF document.
    """

    document_id: str = Field(
        default_factory=lambda: str(uuid4())
    )

    filename: str

    file_size_bytes: int

    content_type: str = "application/pdf"

    page_count: int

    total_characters: int

    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    ingestion_status: str = "text_extracted"

    source_type: str = "uploaded_pdf"

    extraction_method: str = "pypdf"

    metadata_version: str = "1.0"