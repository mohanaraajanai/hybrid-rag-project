"""
Phase 11 end-to-end validation.

Validates:
- Hybrid retrieval
- Graph context inclusion
- Vector context inclusion
- Context statistics
- Answer generation
- Final pipeline response
"""

import json

from app.rag.rag_pipeline import RAGPipeline


def main():
    query = "What technologies are used for databases and storage?"

    print("\n=== PHASE 11 END-TO-END PIPELINE TEST ===")
    print(f"User Query: {query}")

    pipeline = RAGPipeline()

    result = pipeline.run(
        query=query,
        max_context_chunks=8,
    )

    print("\n--- Pipeline Result ---")

    print(
        json.dumps(
            result,
            indent=2,
            default=str,
        )
    )

    assert result["success"] is True, (
        f"Pipeline failed: {result.get('error')}"
    )

    assert result.get("answer"), (
        "Pipeline answer should not be empty."
    )

    retrieval_summary = result.get(
        "retrieval_summary",
        {},
    )

    context_statistics = result.get(
        "context_statistics",
        {},
    )

    debug = result.get(
        "debug",
        {},
    )

    assert retrieval_summary.get(
        "graph_entity_count",
        0,
    ) > 0, (
        "Expected graph entities from retrieval."
    )

    assert retrieval_summary.get(
        "graph_relationship_count",
        0,
    ) > 0, (
        "Expected graph relationships from retrieval."
    )

    assert context_statistics.get(
        "entity_count",
        0,
    ) > 0, (
        "Expected graph entities in context statistics."
    )

    assert context_statistics.get(
        "relationship_count",
        0,
    ) > 0, (
        "Expected graph relationships in context statistics."
    )

    assert debug.get(
        "graph_context_included",
        False,
    ) is True, (
        "Graph context was not included."
    )

    assert debug.get(
        "vector_context_included",
        False,
    ) is True, (
        "Vector context was not included."
    )

    print("\n--- Generated Answer ---")
    print(result["answer"])

    print("\n--- Context Statistics ---")
    print(
        json.dumps(
            context_statistics,
            indent=2,
        )
    )

    print("\n--- Debug Information ---")
    print(
        json.dumps(
            debug,
            indent=2,
        )
    )

    print(
        "\n=== PHASE 11 END-TO-END TEST PASSED ==="
    )


if __name__ == "__main__":
    main()