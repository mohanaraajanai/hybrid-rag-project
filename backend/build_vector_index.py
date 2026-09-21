import json
from pathlib import Path

from app.embeddings.local_embeddings import generate_embeddings
from app.ingestion.chunker import create_document_chunks
from app.vector_store.faiss_store import FAISSVectorStore


# Project root:
# hybrid-rag-project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Processed JSON files location:
# hybrid-rag-project/data/processed/
PROCESSED_DIRECTORY = PROJECT_ROOT / "data" / "processed"


def main() -> None:
    """
    Build a FAISS vector index from the latest processed PDF document.
    """

    print(f"Looking for processed files in: {PROCESSED_DIRECTORY}")

    if not PROCESSED_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Processed directory does not exist: {PROCESSED_DIRECTORY}"
        )

    # Find processed JSON files and sort by latest modification time.
    processed_files = sorted(
        PROCESSED_DIRECTORY.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not processed_files:
        raise FileNotFoundError(
            f"No processed JSON files found in: {PROCESSED_DIRECTORY}"
        )

    # Select the most recently created/updated processed file.
    latest_file = processed_files[0]

    print(f"Loading processed document: {latest_file.name}")

    processed_document = json.loads(
        latest_file.read_text(encoding="utf-8")
    )

    # Create text chunks from page-wise extracted content.
    chunks = create_document_chunks(processed_document)

    if not chunks:
        raise ValueError(
            "No text chunks were created from the processed document."
        )

    print(f"Created {len(chunks)} text chunks.")

    # Extract text from each chunk.
    texts = [chunk["text"] for chunk in chunks]

    # Generate local Hugging Face embeddings.
    print("Generating local embeddings...")
    embeddings = generate_embeddings(texts)

    print(f"Embedding shape: {embeddings.shape}")

    # Create and save the FAISS index.
    vector_store = FAISSVectorStore(index_name="documents")

    vector_store.create_index(embeddings)
    vector_store.add_metadata(chunks)
    vector_store.save()

    print("\nFAISS index saved successfully.")
    print(f"Index location: {vector_store.index_path}")
    print(f"Metadata location: {vector_store.metadata_path}")

    # Test vector search.
    test_query = "What technologies are used in the backend?"

    print(f"\nTesting query: {test_query}")

    query_embedding = generate_embeddings([test_query])

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=3,
    )

    print("\nTop retrieval results:\n")

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(
            f"Result {rank} | "
            f"Score: {result['score']:.4f} | "
            f"Page: {metadata['page_number']}"
        )

        print(metadata["text"][:300])
        print("-" * 60)


if __name__ == "__main__":
    main()