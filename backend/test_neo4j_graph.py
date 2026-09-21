from app.knowledge_graph.neo4j_connection import Neo4jConnection


def main() -> None:
    print("Testing Neo4j graph operations...")

    connection = None

    try:
        connection = Neo4jConnection()

        if not connection.verify_connection():
            print("Neo4j connection failed.")
            return

        print("Neo4j connection successful!")

        # Clear only the test nodes created by this script
        delete_query = """
        MATCH (n:TestTechnology)
        DETACH DELETE n
        """

        connection.execute_query(delete_query)

        # Create two technology nodes and a relationship
        create_query = """
        CREATE (python:TestTechnology {name: 'Python'})
        CREATE (neo4j:TestTechnology {name: 'Neo4j'})
        CREATE (python)-[:USED_WITH]->(neo4j)
        RETURN python.name AS source,
               neo4j.name AS target
        """

        created_records = connection.execute_query(create_query)

        print(f"Created graph data: {created_records}")

        # Read the graph
        read_query = """
        MATCH (source:TestTechnology)-[relationship:USED_WITH]->(target:TestTechnology)
        RETURN source.name AS source,
               type(relationship) AS relationship,
               target.name AS target
        """

        graph_records = connection.execute_query(read_query)

        print("\nRetrieved graph data:")

        for record in graph_records:
            print(record)

    except Exception as error:
        print(f"Error: {error}")

    finally:
        if connection:
            connection.close()
            print("\nNeo4j connection closed.")


if __name__ == "__main__":
    main()