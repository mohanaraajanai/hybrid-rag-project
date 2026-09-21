from pprint import pprint

from app.retrieval.vector_retriever import VectorRetriever


def main():
    retriever = VectorRetriever()

    try:
        results = retriever.retrieve(
            query="What technologies are used in the backend?",
            top_k=2,
        )

        print("\n=== RAW VECTOR RESULT STRUCTURE ===\n")

        for index, result in enumerate(results, start=1):
            print(f"Result {index}:")
            pprint(result)
            print("-" * 60)

    finally:
        # VectorRetriever currently does not require a close operation.
        pass


if __name__ == "__main__":
    main()