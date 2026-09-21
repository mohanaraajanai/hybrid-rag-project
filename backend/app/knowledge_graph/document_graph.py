"""
Document graph storage for Hybrid RAG.

Stores:
1. HybridDocument nodes
2. HybridChunk nodes
3. HAS_HYBRID_CHUNK relationships

The chunk IDs are generated using the same chunker used by
the FAISS vector pipeline.
"""

import json
from pathlib import Path
from typing import Any

from app.ingestion.chunker import create_document_chunks
from app.knowledge_graph.neo4j_connection import Neo4jConnection


class DocumentGraph:
    """
    Stores documents and their text chunks in the isolated
    Hybrid RAG Neo4j graph.
    """

    def __init__(self) -> None:
        self.connection = Neo4jConnection()

    def close(self) -> None:
        """
        Close the Neo4j connection.
        """
        self.connection.close()

    def create_constraints(self) -> None:
        """
        Create uniqueness constraints for Hybrid RAG nodes.
        """

        document_constraint = """
        CREATE CONSTRAINT hybrid_document_id_unique IF NOT EXISTS
        FOR (d:HybridDocument)
        REQUIRE d.document_id IS UNIQUE
        """

        chunk_constraint = """
        CREATE CONSTRAINT hybrid_chunk_id_unique IF NOT EXISTS
        FOR (c:HybridChunk)
        REQUIRE c.chunk_id IS UNIQUE
        """

        self.connection.execute_query(
            document_constraint
        )

        self.connection.execute_query(
            chunk_constraint
        )

    def store_document(
        self,
        processed_file: str,
    ) -> dict[str, Any]:
        """
        Store a processed document and its generated text chunks.

        The chunk IDs are generated using create_document_chunks()
        so that they match the FAISS metadata chunk IDs.
        """

        file_path = Path(processed_file)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Processed file not found: {file_path}"
            )

        with file_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

        document = data["document"]

        document_id = document["document_id"]
        filename = document["filename"]

        # Generate the exact same chunks used by FAISS.
        chunks = create_document_chunks(
            data
        )

        # ---------------------------------------------------------
        # Step 1: Store the document node
        # ---------------------------------------------------------
        document_query = """
        MERGE (d:HybridDocument {
            document_id: $document_id
        })

        SET d.filename = $filename,
            d.file_size_bytes = $file_size_bytes,
            d.page_count = $page_count,
            d.total_characters = $total_characters,
            d.source_type = $source_type,
            d.extraction_method = $extraction_method,
            d.processed_file = $processed_file

        RETURN d.document_id AS document_id
        """

        self.connection.execute_query(
            document_query,
            {
                "document_id": document_id,
                "filename": filename,
                "file_size_bytes": document.get(
                    "file_size_bytes"
                ),
                "page_count": document.get(
                    "page_count"
                ),
                "total_characters": document.get(
                    "total_characters"
                ),
                "source_type": document.get(
                    "source_type"
                ),
                "extraction_method": document.get(
                    "extraction_method"
                ),
                "processed_file": str(file_path),
            },
        )

        # ---------------------------------------------------------
        # Step 2: Store chunks using the shared chunk IDs
        # ---------------------------------------------------------
        chunk_query = """
        MATCH (d:HybridDocument {
            document_id: $document_id
        })

        MERGE (c:HybridChunk {
            chunk_id: $chunk_id
        })

        SET c.document_id = $document_id,
            c.filename = $filename,
            c.page_number = $page_number,
            c.page_chunk_index = $page_chunk_index,
            c.text = $text,
            c.character_count = $character_count,
            c.chunk_type = 'text'

        MERGE (d)-[:HAS_HYBRID_CHUNK]->(c)
        """

        stored_chunks = 0

        for chunk in chunks:
            self.connection.execute_query(
                chunk_query,
                {
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_id": chunk["chunk_id"],
                    "page_number": chunk[
                        "page_number"
                    ],
                    "page_chunk_index": chunk[
                        "page_chunk_index"
                    ],
                    "text": chunk["text"],
                    "character_count": chunk[
                        "character_count"
                    ],
                },
            )

            stored_chunks += 1

        return {
            "document_id": document_id,
            "filename": filename,
            "chunks_stored": stored_chunks,
        }