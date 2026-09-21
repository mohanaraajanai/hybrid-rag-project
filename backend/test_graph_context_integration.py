"""
Phase 11:
Validate graph data integration into the final context.
"""

import json

from app.retrieval.hybrid_retriever import HybridRetriever
from app.rag.context_builder import ContextBuilder


def main():
    query = "What technologies are used in the backend?"

    print("\n=== PHASE 11: GRAPH CONTEXT INTEGRATION TEST ===")
    print(f"Query: {query}")

    retriever = HybridRetriever()

    hybrid_result = retriever.retrieve(query)

    print("\n--- Top-Level Hybrid Result Keys ---")
    print(list(hybrid_result.keys()))

    print("\n--- Retrieval Summary ---")
    print(
        json.dumps(
            hybrid_result.get("summary", {}),
            indent=2,
            default=str,
        )
    )

    context_builder = ContextBuilder(
        max_context_chunks=8,
    )

    context = context_builder.build_context(
        query=query,
        hybrid_results=hybrid_result,
    )

    print("\n--- Context Preview ---")
    print(context[:15000])

    stats = context_builder.get_context_statistics(
        hybrid_result=hybrid_result,
        context=context,
    )

    print("\n--- Context Statistics ---")
    print(
        json.dumps(
            stats,
            indent=2,
            default=str,
        )
    )

    assert context.strip(), (
        "Context should not be empty."
    )

    assert "KNOWLEDGE GRAPH ENTITIES" in context, (
        "Graph entity section missing from context."
    )

    assert "KNOWLEDGE GRAPH RELATIONSHIPS" in context, (
        "Graph relationship section missing from context."
    )

    assert stats["entity_count"] > 0, (
        "Graph entity count should be greater than zero."
    )

    assert stats["relationship_count"] > 0, (
        "Graph relationship count should be greater than zero."
    )

    assert stats["graph_context_included"] is True, (
        "Graph context should be marked as included."
    )

    assert stats["vector_context_included"] is True, (
        "Vector context should be marked as included."
    )

    print(
        "\n=== GRAPH CONTEXT INTEGRATION TEST PASSED ==="
    )

    retriever.close()


if __name__ == "__main__":
    main()