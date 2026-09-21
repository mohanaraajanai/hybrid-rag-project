import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable


# Load the project's .env file
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(dotenv_path=ENV_PATH)


class Neo4jConnection:
    """
    Handles connection and queries to the Neo4j database.
    """

    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.username = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")

        if not self.uri:
            raise ValueError("NEO4J_URI is missing in the .env file.")

        if not self.username:
            raise ValueError("NEO4J_USERNAME is missing in the .env file.")

        if not self.password:
            raise ValueError("NEO4J_PASSWORD is missing in the .env file.")

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.username, self.password),
        )

    def verify_connection(self) -> bool:
        """
        Verify that the Neo4j database is reachable.
        """
        try:
            self.driver.verify_connectivity()
            return True
        except ServiceUnavailable as error:
            print(f"Neo4j connection failed: {error}")
            return False

    def execute_query(self, query: str, parameters: dict | None = None):
        """
        Execute a Cypher query and return the records.
        """
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return result.data()

    def close(self):
        """
        Close the Neo4j driver connection.
        """
        self.driver.close()