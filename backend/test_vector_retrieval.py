from app.retrieval.vector_retriever import VectorRetriever


def main() -> None:
    """
    Test the vector retrieval service.
    """

    print("Loading FAISS vector retriever...")

    retriever = VectorRetriever(
        index_name="documents"
    )

    query = "What technologies are used in the backend?"

    print(f"\nQuery: {query}")
    print("\nRetrieved results:\n")

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        metadata = result["metadata"]

        print(f"Result {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Document: {metadata['filename']}")
        print(f"Page: {metadata['page_number']}")
        print(f"Chunk ID: {metadata['chunk_id']}")
        print(f"Text:\n{metadata['text']}")
        print("-" * 70)


if __name__ == "__main__":
    main()