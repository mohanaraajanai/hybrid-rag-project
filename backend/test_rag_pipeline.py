"""
Test the complete End-to-End Hybrid RAG Pipeline.
"""

import json

from app.rag.rag_pipeline import RAGPipeline


def main():
    print("\n=== END-TO-END HYBRID RAG PIPELINE TEST ===\n")

    query = "What technologies are used for databases and storage?"

    print(f"User Query: {query}")
    print("\nRunning pipeline...\n")

    pipeline = RAGPipeline()

    result = pipeline.run(
        query=query,
        max_context_chunks=5,
    )

    print("Generated Answer:")
    print("-----------------")
    print(result.get("answer", "No answer generated."))

    print("\nSources:")
    print("--------")

    for index, source in enumerate(
        result.get("sources", []),
        start=1,
    ):
        print(f"{index}. {source}")

    print("\nRetrieval Summary:")
    print("------------------")
    print(
        json.dumps(
            result.get("retrieval_summary", {}),
            indent=2,
        )
    )

    print("\nContext Statistics:")
    print("-------------------")
    print(
        json.dumps(
            result.get("context_statistics", {}),
            indent=2,
        )
    )

    print("\nTimings:")
    print("--------")
    print(
        json.dumps(
            result.get("timings", {}),
            indent=2,
        )
    )

    print("\nGuardrail Status:")
    print("-----------------")
    print(result.get("guardrail_status"))

    if not result.get("success"):
        print("\nPipeline Error:")
        print(result.get("error"))

        raise RuntimeError(
            "End-to-end pipeline test failed."
        )

    if not result.get("answer"):
        raise RuntimeError(
            "Pipeline returned an empty answer."
        )

    print("\n=== END-TO-END PIPELINE TEST PASSED ===")


if __name__ == "__main__":
    main()