from app.knowledge_graph.entity_extractor import EntityExtractor
from app.knowledge_graph.entity_graph import EntityGraph
from app.knowledge_graph.neo4j_connection import Neo4jConnection


def main() -> None:
    print("Testing complete entity extraction and graph storage pipeline...")

    connection = Neo4jConnection()
    entity_graph = EntityGraph()

    try:
        if not connection.verify_connection():
            print("Neo4j connection failed.")
            return

        print("Neo4j connection successful!")

        entity_graph.create_constraints()
        print("Entity constraint verified.")

        # Retrieve one HybridChunk
        chunk_result = connection.execute_query(
            """
            MATCH (c:HybridChunk)
            RETURN c.chunk_id AS chunk_id,
                   c.text AS text
            ORDER BY c.page_number
            LIMIT 1
            """
        )

        if not chunk_result:
            print("No HybridChunk found.")
            return

        chunk = chunk_result[0]

        print(f"Using chunk: {chunk['chunk_id']}")
        print("Calling LLM for extraction...")

        extractor = EntityExtractor()
        extraction_result = extractor.extract(chunk["text"])

        print(
            f"Extracted {len(extraction_result['entities'])} entities "
            f"and {len(extraction_result['relationships'])} relationships."
        )

        # Store the extracted data in Neo4j
        storage_result = entity_graph.store_extraction(
            chunk_id=chunk["chunk_id"],
            extraction=extraction_result,
        )

        print("\nGraph storage result:")
        print(storage_result)

        print("\nComplete entity pipeline finished successfully.")

    except Exception as error:
        print(f"Error: {error}")

    finally:
        entity_graph.close()
        connection.close()
        print("\nNeo4j connections closed.")


if __name__ == "__main__":
    main()