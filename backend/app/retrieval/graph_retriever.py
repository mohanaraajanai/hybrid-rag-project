import json
import re
from pathlib import Path
from typing import Any

from app.knowledge_graph.neo4j_connection import Neo4jConnection


PROJECT_ROOT = Path(__file__).resolve().parents[3]

VECTOR_METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "vector_store"
    / "documents_metadata.json"
)


class GraphRetriever:
    """
    Retrieves graph context for the same document corpus represented
    by the active FAISS vector index.

    The FAISS metadata file is used to determine which document IDs
    should be searched in Neo4j. This prevents older documents already
    stored in Neo4j from contaminating the current Hybrid RAG result.
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

    def __init__(self) -> None:
        self.connection = Neo4jConnection()

    def close(self) -> None:
        self.connection.close()

    def extract_terms(self, query: str) -> list[str]:
        """Extract searchable keywords from the user query."""

        if not query or not query.strip():
            raise ValueError("Graph retrieval query cannot be empty.")

        words = re.findall(
            r"[a-zA-Z0-9+#.-]+",
            query.lower(),
        )

        terms = [
            word
            for word in words
            if len(word) >= 3
            and word not in self.STOPWORDS
        ]

        return list(dict.fromkeys(terms))

    def _get_active_document_ids(self) -> list[str]:
        """
        Read document IDs represented by the active FAISS metadata.

        The current FAISS build represents the document corpus used by
        vector retrieval. Graph retrieval should use the same corpus.
        """

        if not VECTOR_METADATA_PATH.exists():
            return []

        try:
            metadata = json.loads(
                VECTOR_METADATA_PATH.read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError):
            return []

        document_ids = {
            str(item.get("document_id"))
            for item in metadata
            if isinstance(item, dict)
            and item.get("document_id")
        }

        return sorted(document_ids)

    def _format_records(
        self,
        records: list[dict[str, Any]],
        query: str,
        terms: list[str],
    ) -> dict[str, Any]:
        """Convert Neo4j records into a consistent response structure."""

        entities: list[dict[str, Any]] = []
        relationships: list[dict[str, Any]] = []
        chunks_by_id: dict[str, dict[str, Any]] = {}

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

            for relationship in (
                record.get("relationships") or []
            ):
                related_entity = relationship.get(
                    "related_entity"
                )

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

            for chunk in record.get("chunks") or []:
                chunk_id = chunk.get("chunk_id")

                if not chunk_id:
                    continue

                if chunk_id in chunks_by_id:
                    continue

                chunks_by_id[chunk_id] = {
                    "chunk_id": chunk_id,
                    "document_id": chunk.get(
                        "document_id"
                    ),
                    "filename": chunk.get(
                        "filename"
                    ),
                    "page_number": chunk.get(
                        "page_number"
                    ),
                    "page_chunk_index": chunk.get(
                        "page_chunk_index"
                    ),
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
        Retrieve graph context restricted to the documents represented
        by the active FAISS metadata.
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

        active_document_ids = (
            self._get_active_document_ids()
        )

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

        AND (
            size($document_ids) = 0
            OR EXISTS {
                MATCH (scoped_chunk:HybridChunk)
                    -[:MENTIONS_HYBRID_ENTITY]->(entity)
                WHERE scoped_chunk.document_id IN $document_ids
            }
        )

        OPTIONAL MATCH
            (chunk:HybridChunk)
            -[:MENTIONS_HYBRID_ENTITY]->(entity)

        WHERE
            size($document_ids) = 0
            OR chunk.document_id IN $document_ids

        OPTIONAL MATCH
            (entity)-[relationship]-(related:HybridEntity)

        WHERE
            size($document_ids) = 0
            OR EXISTS {
                MATCH (related_chunk:HybridChunk)
                    -[:MENTIONS_HYBRID_ENTITY]->(related)
                WHERE related_chunk.document_id IN $document_ids
            }

        RETURN
            entity.entity_id AS entity_id,
            entity.name AS entity_name,
            entity.entity_type AS entity_type,

            collect(DISTINCT {
                chunk_id: chunk.chunk_id,
                document_id: chunk.document_id,
                filename: chunk.filename,
                page_number: chunk.page_number,
                page_chunk_index: chunk.page_chunk_index,
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
                "document_ids": active_document_ids,
                "max_entities": max_entities,
            },
        )

        result = self._format_records(
            records,
            query,
            terms,
        )

        if result["entities"]:
            result["chunks"] = (
                result["chunks"][:max_chunks]
            )

            return result

        # --------------------------------------------------
        # Strategy 2: Scoped fallback graph retrieval
        # --------------------------------------------------

        fallback_query = """
        MATCH (chunk:HybridChunk)

        WHERE
            size($document_ids) = 0
            OR chunk.document_id IN $document_ids

        OPTIONAL MATCH
            (chunk)-[:MENTIONS_HYBRID_ENTITY]->(entity:HybridEntity)

        OPTIONAL MATCH
            (entity)-[relationship]-(related:HybridEntity)

        WHERE
            size($document_ids) = 0
            OR related IS NULL
            OR EXISTS {
                MATCH (related_chunk:HybridChunk)
                    -[:MENTIONS_HYBRID_ENTITY]->(related)
                WHERE related_chunk.document_id IN $document_ids
            }

        RETURN
            entity.entity_id AS entity_id,
            entity.name AS entity_name,
            entity.entity_type AS entity_type,

            collect(DISTINCT {
                chunk_id: chunk.chunk_id,
                document_id: chunk.document_id,
                filename: chunk.filename,
                page_number: chunk.page_number,
                page_chunk_index: chunk.page_chunk_index,
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
                "document_ids": active_document_ids,
                "max_entities": max_entities,
            },
        )

        fallback_result = self._format_records(
            fallback_records,
            query,
            terms,
        )

        fallback_result["chunks"] = (
            fallback_result["chunks"][:max_chunks]
        )

        return fallback_result
