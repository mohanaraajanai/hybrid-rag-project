import re
from typing import Any

from app.knowledge_graph.neo4j_connection import Neo4jConnection


class GraphRetriever:
    """
    Retrieves entities, relationships, and source chunks
    from the isolated Hybrid RAG knowledge graph.
    """

    STOPWORDS = {
        "what",
        "which",
        "where",
        "when",
        "who",
        "why",
        "how",
        "are",
        "is",
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "in",
        "on",
        "for",
        "to",
        "used",
        "use",
        "with",
        "does",
        "do",
        "this",
        "that",
        "these",
        "those",
    }

    def __init__(self):
        self.connection = Neo4jConnection()

    def close(self):
        self.connection.close()

    def extract_terms(self, query: str) -> list[str]:
        """Extract searchable keywords from the user query."""

        if not query or not query.strip():
            raise ValueError("Graph retrieval query cannot be empty.")

        words = re.findall(r"[a-zA-Z0-9+#.-]+", query.lower())

        terms = [
            word
            for word in words
            if len(word) >= 3 and word not in self.STOPWORDS
        ]

        return list(dict.fromkeys(terms))

    def _format_records(
        self,
        records: list[dict[str, Any]],
        query: str,
        terms: list[str],
    ) -> dict[str, Any]:
        """Convert Neo4j records into a consistent response structure."""

        entities = []
        relationships = []
        chunks_by_id = {}

        for record in records:
            entity_name = record.get("entity_name")

            if not entity_name:
                continue

            entities.append(
                {
                    "entity_id": record.get("entity_id"),
                    "name": entity_name,
                    "type": record.get("entity_type"),
                }
            )

            for relationship in record.get("relationships", []):
                related_entity = relationship.get("related_entity")

                if related_entity:
                    relationships.append(
                        {
                            "source": entity_name,
                            "target": related_entity,
                            "relationship": relationship.get(
                                "relationship"
                            ),
                        }
                    )

            for chunk in record.get("chunks", []):
                chunk_id = chunk.get("chunk_id")

                if chunk_id and chunk_id not in chunks_by_id:
                    chunks_by_id[chunk_id] = {
                        "chunk_id": chunk_id,
                        "page_number": chunk.get("page_number"),
                        "text": chunk.get("text"),
                    }

        return {
            "query": query,
            "search_terms": terms,
            "entities": entities,
            "relationships": relationships,
            "chunks": list(chunks_by_id.values()),
        }

    def retrieve(
        self,
        query: str,
        max_entities: int = 15,
        max_chunks: int = 10,
    ) -> dict[str, Any]:
        """
        Retrieve graph context.

        First attempts keyword-based entity matching.
        If no entities are found, falls back to retrieving
        entities from the stored HybridDocument graph.
        """

        terms = self.extract_terms(query)

        if not terms:
            return {
                "query": query,
                "search_terms": [],
                "entities": [],
                "relationships": [],
                "chunks": [],
            }

        # --------------------------------------------------
        # Strategy 1: Keyword-based entity retrieval
        # --------------------------------------------------

        entity_query = """
        MATCH (entity:HybridEntity)
        WHERE any(
            term IN $terms
            WHERE
                toLower(entity.name) CONTAINS term
                OR term CONTAINS toLower(entity.name)
        )

        OPTIONAL MATCH (chunk:HybridChunk)-[:MENTIONS_HYBRID_ENTITY]->(entity)

        OPTIONAL MATCH (entity)-[relationship]-(related:HybridEntity)

        RETURN
            entity.entity_id AS entity_id,
            entity.name AS entity_name,
            entity.entity_type AS entity_type,

            collect(DISTINCT {
                chunk_id: chunk.chunk_id,
                page_number: chunk.page_number,
                text: chunk.text
            }) AS chunks,

            collect(DISTINCT {
                related_entity: related.name,
                related_entity_type: related.entity_type,
                relationship: type(relationship)
            }) AS relationships

        LIMIT $max_entities
        """

        records = self.connection.execute_query(
            entity_query,
            {
                "terms": terms,
                "max_entities": max_entities,
            },
        )

        result = self._format_records(records, query, terms)

        if result["entities"]:
            result["chunks"] = result["chunks"][:max_chunks]
            return result

        # --------------------------------------------------
        # Strategy 2: Fallback graph retrieval
        # --------------------------------------------------

        fallback_query = """
        MATCH (chunk:HybridChunk)
        OPTIONAL MATCH (chunk)-[:MENTIONS_HYBRID_ENTITY]->(entity:HybridEntity)
        OPTIONAL MATCH (entity)-[relationship]-(related:HybridEntity)

        RETURN
            entity.entity_id AS entity_id,
            entity.name AS entity_name,
            entity.entity_type AS entity_type,

            collect(DISTINCT {
                chunk_id: chunk.chunk_id,
                page_number: chunk.page_number,
                text: chunk.text
            }) AS chunks,

            collect(DISTINCT {
                related_entity: related.name,
                related_entity_type: related.entity_type,
                relationship: type(relationship)
            }) AS relationships

        LIMIT $max_entities
        """

        fallback_records = self.connection.execute_query(
            fallback_query,
            {
                "max_entities": max_entities,
            },
        )

        fallback_result = self._format_records(
            fallback_records,
            query,
            terms,
        )

        fallback_result["chunks"] = fallback_result["chunks"][:max_chunks]

        return fallback_result