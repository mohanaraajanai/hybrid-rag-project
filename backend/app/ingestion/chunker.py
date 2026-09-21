from typing import Any


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150


def split_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """
    Split text into overlapping character-based chunks.
    """

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size."
        )

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - chunk_overlap

    return chunks


def create_document_chunks(
    processed_document: dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    """
    Create chunks from page-wise extracted PDF text.

    Each chunk retains document and page metadata.
    """

    document_metadata = processed_document["document"]
    extraction = processed_document["extraction"]

    document_id = document_metadata["document_id"]
    filename = document_metadata["filename"]

    document_chunks = []
    chunk_id = 0

    for page in extraction["pages"]:
        page_number = page["page_number"]
        page_text = page["text"]

        page_chunks = split_text(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for page_chunk_index, chunk_text in enumerate(page_chunks):
            document_chunks.append(
                {
                    "chunk_id": f"{document_id}_{chunk_id}",
                    "document_id": document_id,
                    "filename": filename,
                    "page_number": page_number,
                    "page_chunk_index": page_chunk_index,
                    "text": chunk_text,
                    "character_count": len(chunk_text),
                }
            )

            chunk_id += 1

    return document_chunks