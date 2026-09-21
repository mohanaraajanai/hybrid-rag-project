import re
from typing import Any

from app.knowledge_graph.neo4j_connection import Neo4jConnection


class EntityGraph:
    """
    Stores extracted entities and relationships in the isolated
    Hybrid RAG graph.
    """

    ALLOWED_RELATIONSHIPS = {
        "USES",
        "PART_OF",
        "PROVIDES",
        "DEPENDS_ON",
        "RELATED_TO",
        "OTHER",
    }

    def __init__(self):
        self.connection = Neo4jConnection()

    def close(self) -> None:
        self.connection.close()

    def create_constraints(self) -> None:
        query = """
        CREATE CONSTRAINT hybrid_entity_unique IF NOT EXISTS
        FOR (e:HybridEntity)
        REQUIRE e.entity_id IS UNIQUE
        """

        self.connection.execute_query(query)

    @staticmethod
    def create_entity_id(entity_name: str, entity_type: str) -> str:
        normalized_name = entity_name.strip().lower()
        normalized_type = entity_type.strip().lower()

        return f"{normalized_type}::{normalized_name}"

    @staticmethod
    def normalize_relationship_type(relationship: str) -> str:
        relationship = relationship.strip().upper()

        if relationship not in EntityGraph.ALLOWED_RELATIONSHIPS:
            return "OTHER"

        return relationship

    def store_extraction(
        self,
        chunk_id: str,
        extraction: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Store all entities and relationships extracted from one chunk.
        """

        entities = extraction.get("entities", [])
        relationships = extraction.get("relationships", [])

        stored_entities = 0
        stored_relationships = 0

        entity_lookup: dict[str, str] = {}

        # 1. Store entities and link them to the source chunk
        entity_query = """
        MATCH (c:HybridChunk {chunk_id: $chunk_id})

        MERGE (e:HybridEntity {entity_id: $entity_id})
        SET e.name = $entity_name,
            e.entity_type = $entity_type

        MERGE (c)-[:MENTIONS_HYBRID_ENTITY]->(e)
        """

        for entity in entities:
            name = entity.get("name", "").strip()
            entity_type = entity.get("type", "Other").strip()

            if not name:
                continue

            entity_id = self.create_entity_id(name, entity_type)

            self.connection.execute_query(
                entity_query,
                {
                    "chunk_id": chunk_id,
                    "entity_id": entity_id,
                    "entity_name": name,
                    "entity_type": entity_type,
                },
            )

            entity_lookup[name.lower()] = entity_id
            stored_entities += 1

        # 2. Store entity-to-entity relationships
        for relationship in relationships:
            source = relationship.get("source", "").strip()
            target = relationship.get("target", "").strip()
            relation_type = self.normalize_relationship_type(
                relationship.get("relationship", "RELATED_TO")
            )

            if not source or not target:
                continue

            source_id = entity_lookup.get(source.lower())
            target_id = entity_lookup.get(target.lower())

            if not source_id or not target_id:
                continue

            relationship_query = f"""
            MATCH (source:HybridEntity {{entity_id: $source_id}})
            MATCH (target:HybridEntity {{entity_id: $target_id}})
            MERGE (source)-[:{relation_type}]->(target)
            """

            self.connection.execute_query(
                relationship_query,
                {
                    "source_id": source_id,
                    "target_id": target_id,
                },
            )

            stored_relationships += 1

        return {
            "chunk_id": chunk_id,
            "entities_stored": stored_entities,
            "relationships_stored": stored_relationships,
        }

    def store_entity(
        self,
        chunk_id: str,
        entity_name: str,
        entity_type: str,
    ) -> dict[str, Any]:
        """
        Store one entity.
        Retained for individual entity testing.
        """

        entity_id = self.create_entity_id(
            entity_name,
            entity_type,
        )

        query = """
        MATCH (c:HybridChunk {chunk_id: $chunk_id})

        MERGE (e:HybridEntity {entity_id: $entity_id})
        SET e.name = $entity_name,
            e.entity_type = $entity_type

        MERGE (c)-[:MENTIONS_HYBRID_ENTITY]->(e)

        RETURN e.entity_id AS entity_id,
               e.name AS name,
               e.entity_type AS entity_type
        """

        result = self.connection.execute_query(
            query,
            {
                "chunk_id": chunk_id,
                "entity_id": entity_id,
                "entity_name": entity_name,
                "entity_type": entity_type,
            },
        )

        return result[0] if result else {}