from app.knowledge_graph.entity_graph import EntityGraph
from app.knowledge_graph.neo4j_connection import Neo4jConnection


def main() -> None:
    print("Testing entity graph storage...")

    connection = Neo4jConnection()
    entity_graph = EntityGraph()

    try:
        if not connection.verify_connection():
            print("Neo4j connection failed.")
            return

        print("Neo4j connection successful!")

        entity_graph.create_constraints()
        print("Entity constraint created.")

        chunk_result = connection.execute_query(
            """
            MATCH (c:HybridChunk)
            RETURN c.chunk_id AS chunk_id
            ORDER BY c.page_number
            LIMIT 1
            """
        )

        if not chunk_result:
            print("No HybridChunk found.")
            return

        chunk_id = chunk_result[0]["chunk_id"]

        print(f"Using chunk: {chunk_id}")

        entity_result = entity_graph.store_entity(
            chunk_id=chunk_id,
            entity_name="Node.js",
            entity_type="Technology",
        )

        print("\nEntity stored successfully:")
        print(entity_result)

    except Exception as error:
        print(f"Error: {error}")

    finally:
        entity_graph.close()
        connection.close()
        print("\nNeo4j connections closed.")


if __name__ == "__main__":
    main()