from pathlib import Path

from app.knowledge_graph.document_graph import DocumentGraph


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    processed_dir = project_root / "data" / "processed"
    processed_files = list(processed_dir.glob("*.json"))

    if not processed_files:
        print("No processed JSON files found.")
        return

    processed_file = processed_files[-1]

    print(f"Using processed file: {processed_file}")

    graph = DocumentGraph()

    try:
        if not graph.connection.verify_connection():
            print("Neo4j connection failed.")
            return

        print("Neo4j connection successful!")

        graph.create_constraints()
        print("Neo4j constraints created.")

        result = graph.store_document(str(processed_file))

        print("\nDocument graph stored successfully:")
        print(result)

    except Exception as error:
        print(f"Error: {error}")

    finally:
        graph.close()
        print("\nNeo4j connection closed.")


if __name__ == "__main__":
    main()