from pathlib import Path
from typing import Any

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> dict[str, Any]:
    """
    Extract text from a PDF file page by page.
    """

    reader = PdfReader(str(pdf_path))

    pages = []
    total_characters = 0

    for page_number, page in enumerate(reader.pages, start=1):
        extracted_text = page.extract_text() or ""

        page_data = {
            "page_number": page_number,
            "text": extracted_text,
            "character_count": len(extracted_text),
        }

        pages.append(page_data)
        total_characters += len(extracted_text)

    return {
        "page_count": len(pages),
        "total_characters": total_characters,
        "extraction_method": "pypdf",
        "pages": pages,
    }