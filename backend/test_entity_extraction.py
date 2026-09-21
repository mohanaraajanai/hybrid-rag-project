from app.knowledge_graph.entity_extractor import EntityExtractor
from app.knowledge_graph.neo4j_connection import Neo4jConnection


def main() -> None:
    print("Testing LLM entity extraction...")

    connection = Neo4jConnection()
    extractor = None

    try:
        if not connection.verify_connection():
            print("Neo4j connection failed.")
            return

        chunk_result = connection.execute_query(
            """
            MATCH (c:HybridChunk)
            RETURN c.text AS text, c.chunk_id AS chunk_id
            ORDER BY c.page_number
            LIMIT 1
            """
        )

        if not chunk_result:
            print("No HybridChunk found.")
            return

        chunk = chunk_result[0]

        print(f"Using chunk: {chunk['chunk_id']}")
        print("Calling the LLM for entity extraction...")

        extractor = EntityExtractor()
        result = extractor.extract(chunk["text"])

        print("\nExtracted entities:")

        for entity in result["entities"]:
            print(
                f"- {entity['name']} "
                f"({entity['type']})"
            )

        print("\nExtracted relationships:")

        for relationship in result["relationships"]:
            print(
                f"- {relationship['source']} "
                f"--{relationship['relationship']}--> "
                f"{relationship['target']}"
            )

        print("\nExtraction completed successfully.")

    except Exception as error:
        print(f"Error: {error}")

    finally:
        connection.close()
        print("\nNeo4j connection closed.")


if __name__ == "__main__":
    main()