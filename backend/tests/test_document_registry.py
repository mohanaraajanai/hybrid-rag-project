from app.document_registry import (
    calculate_file_hash,
    get_document_by_hash,
    is_document_registered,
    list_registered_documents,
    register_document,
    remove_document,
)


def test_document_registry():
    """
    Test duplicate detection and registry operations.
    """

    sample_content = b"Hybrid RAG test document"

    file_hash = calculate_file_hash(sample_content)

    assert len(file_hash) == 64
    assert is_document_registered(file_hash) is False

    document_data = {
        "document_id": "test-document-001",
        "filename": "test_document.pdf",
        "status": "ready",
    }

    registered_document = register_document(
        file_hash=file_hash,
        document_data=document_data,
    )

    assert registered_document["document_id"] == "test-document-001"
    assert registered_document["file_hash"] == file_hash

    assert is_document_registered(file_hash) is True

    retrieved_document = get_document_by_hash(file_hash)

    assert retrieved_document is not None
    assert retrieved_document["filename"] == "test_document.pdf"

    duplicate_result = register_document(
        file_hash=file_hash,
        document_data={
            "document_id": "different-document-id",
            "filename": "duplicate.pdf",
            "status": "ready",
        },
    )

    # The original record must be returned.
    assert duplicate_result["document_id"] == "test-document-001"

    documents = list_registered_documents()

    assert len(documents) >= 1

    removed = remove_document(file_hash)

    assert removed is True
    assert is_document_registered(file_hash) is False