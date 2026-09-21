from app.knowledge_graph.neo4j_connection import Neo4jConnection


def main() -> None:
    print("Testing Neo4j connection...")

    connection = None

    try:
        connection = Neo4jConnection()

        if connection.verify_connection():
            print("Neo4j connection successful!")

            result = connection.execute_query(
                "RETURN 'Neo4j is working!' AS message"
            )

            print(f"Database response: {result}")

        else:
            print("Neo4j connection failed.")

    except Exception as error:
        print(f"Error: {error}")

    finally:
        if connection:
            connection.close()
            print("Neo4j connection closed.")


if __name__ == "__main__":
    main()