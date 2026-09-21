from pprint import pprint

from app.retrieval.hybrid_retriever import HybridRetriever
from app.rag.context_builder import ContextBuilder


def main():
    query = "What technologies are used in the backend?"

    hybrid_retriever = HybridRetriever()
    context_builder = ContextBuilder()

    try:
        # Retrieve information from FAISS and Neo4j
        hybrid_result = hybrid_retriever.retrieve(
            query=query,
            vector_top_k=5,
            graph_max_entities=15,
            graph_max_chunks=10,
        )

        # Build LLM-ready context
        context = context_builder.build_context(
            hybrid_result=hybrid_result,
            max_context_chunks=8,
        )

        print("\n=== GENERATED CONTEXT ===\n")
        print(context)

        print("\n=== CONTEXT BUILDER TEST PASSED ===")

    finally:
        hybrid_retriever.close()


if __name__ == "__main__":
    main()