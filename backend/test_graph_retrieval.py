from pprint import pprint

from app.retrieval.graph_retriever import GraphRetriever


def main():
    query = "What technologies are used in the backend?"

    retriever = GraphRetriever()

    try:
        result = retriever.retrieve(
            query=query,
            max_entities=10,
            max_chunks=10,
        )

        print("\n=== GRAPH RETRIEVAL RESULT ===")

        print("\nQuery:")
        print(result["query"])

        print("\nSearch Terms:")
        print(result["search_terms"])

        print("\nEntities:")
        pprint(result["entities"])

        print("\nRelationships:")
        pprint(result["relationships"])

        print("\nSource Chunks:")
        for chunk in result["chunks"]:
            print(
                f"\nChunk ID: {chunk['chunk_id']}"
                f"\nPage: {chunk['page_number']}"
                f"\nText Preview: {chunk['text'][:300]}..."
            )

    finally:
        retriever.close()


if __name__ == "__main__":
    main()