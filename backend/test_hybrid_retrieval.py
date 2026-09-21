from pprint import pprint

from app.retrieval.hybrid_retriever import HybridRetriever


def main():
    """
    Test the hybrid retrieval pipeline.

    This test retrieves information from:

    1. FAISS vector database
    2. Neo4j knowledge graph
    3. Combined hybrid retrieval output
    """

    query = "What technologies are used in the backend?"

    hybrid_retriever = HybridRetriever()

    try:
        # ---------------------------------------------
        # Execute hybrid retrieval
        # ---------------------------------------------

        result_data = hybrid_retriever.retrieve(
            query=query,
            vector_top_k=5,
            graph_max_entities=15,
            graph_max_chunks=10,
        )

        # ---------------------------------------------
        # Display main result
        # ---------------------------------------------

        print("\n=== HYBRID RETRIEVAL RESULT ===")

        print("\nQuery:")
        print(result_data["query"])

        # ---------------------------------------------
        # Display summary
        # ---------------------------------------------

        print("\nSummary:")
        pprint(result_data["summary"])

        # ---------------------------------------------
        # Display vector retrieval results
        # ---------------------------------------------

        print("\nVector Results:")

        vector_results = result_data.get(
            "vector_results",
            [],
        )

        if not vector_results:
            print("- No vector results found.")

        for result in vector_results:
            metadata = result.get("metadata", {})

            print(
                f"- Chunk: {metadata.get('chunk_id')} "
                f"| Page: {metadata.get('page_number')} "
                f"| Score: {result.get('score')}"
            )

        # ---------------------------------------------
        # Display graph entities
        # ---------------------------------------------

        print("\nGraph Entities:")

        graph_results = result_data.get(
            "graph_results",
            {},
        )

        entities = graph_results.get(
            "entities",
            [],
        )

        if not entities:
            print("- No graph entities found.")

        for entity in entities:
            print(
                f"- {entity.get('name')} "
                f"({entity.get('type')})"
            )

        # ---------------------------------------------
        # Display graph relationships
        # ---------------------------------------------

        print("\nGraph Relationships:")

        relationships = graph_results.get(
            "relationships",
            [],
        )

        if not relationships:
            print("- No graph relationships found.")

        for relationship in relationships:
            print(
                f"- {relationship.get('source')} "
                f"--[{relationship.get('relationship')}]--> "
                f"{relationship.get('target')}"
            )

        # ---------------------------------------------
        # Display combined chunks
        # ---------------------------------------------

        print("\nCombined Chunks:")

        combined_chunks = result_data.get(
            "combined_chunks",
            [],
        )

        if not combined_chunks:
            print("- No combined chunks found.")

        for chunk in combined_chunks:
            print(
                f"- {chunk.get('chunk_id')} "
                f"| Page: {chunk.get('page_number')} "
                f"| Sources: {chunk.get('retrieval_sources')} "
                f"| Vector Score: {chunk.get('vector_score')}"
            )

    finally:
        # ---------------------------------------------
        # Close Neo4j connection
        # ---------------------------------------------

        hybrid_retriever.close()

        print("\nNeo4j connection closed.")


if __name__ == "__main__":
    main()