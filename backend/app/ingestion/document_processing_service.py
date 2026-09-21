"""
Complete Hybrid RAG document processing service.

This service connects:

1. Processed PDF JSON
2. Text chunking
3. Local embeddings
4. FAISS vector storage
5. Neo4j document graph
6. LLM entity extraction
7. Neo4j entity graph

The service is intentionally separate from the upload pipeline so that
the existing upload and duplicate-detection functionality remains stable.
"""

from pathlib import Path
from typing import Any

from app.embeddings.local_embeddings import generate_embeddings
from app.ingestion.chunker import create_document_chunks
from app.knowledge_graph.document_graph import DocumentGraph
from app.knowledge_graph.entity_extractor import EntityExtractor
from app.knowledge_graph.entity_graph import EntityGraph
from app.vector_store.faiss_store import FAISSVectorStore


class DocumentProcessingService:
    """
    Orchestrates all downstream processing for one extracted document.
    """

    def __init__(
        self,
        vector_index_name: str = "documents",
    ) -> None:
        self.vector_index_name = vector_index_name

        self.entity_extractor = EntityExtractor()
        self.document_graph = DocumentGraph()
        self.entity_graph = EntityGraph()
        self.vector_store = FAISSVectorStore(
            index_name=vector_index_name
        )

    def close(self) -> None:
        """
        Close Neo4j connections.
        """
        self.document_graph.close()
        self.entity_graph.close()

    def initialize_graph_constraints(self) -> None:
        """
        Create all required isolated Hybrid RAG constraints.
        """
        self.document_graph.create_constraints()
        self.entity_graph.create_constraints()

    def process_document(
        self,
        processed_file: str | Path,
    ) -> dict[str, Any]:
        """
        Process one extracted document through the complete pipeline.

        Args:
            processed_file:
                Path to the processed PDF JSON file.

        Returns:
            Processing summary.
        """

        processed_file_path = Path(processed_file)

        if not processed_file_path.exists():
            raise FileNotFoundError(
                f"Processed document not found: {processed_file_path}"
            )

        # ---------------------------------------------------------
        # Step 1: Load and chunk the processed document
        # ---------------------------------------------------------
        import json

        processed_document = json.loads(
            processed_file_path.read_text(
                encoding="utf-8"
            )
        )

        document_metadata = processed_document["document"]
        document_id = document_metadata["document_id"]
        filename = document_metadata["filename"]

        chunks = create_document_chunks(
            processed_document
        )

        if not chunks:
            raise ValueError(
                "No text chunks were generated from the document."
            )

        # ---------------------------------------------------------
        # Step 2: Generate local embeddings
        # ---------------------------------------------------------
        chunk_texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = generate_embeddings(
            chunk_texts
        )

        # ---------------------------------------------------------
        # Step 3: Store vector index
        # ---------------------------------------------------------
        self.vector_store.create_index(
            embeddings
        )

        self.vector_store.add_metadata(
            chunks
        )

        self.vector_store.save()

        # ---------------------------------------------------------
        # Step 4: Store document and page information in Neo4j
        # ---------------------------------------------------------
        document_graph_result = (
            self.document_graph.store_document(
                str(processed_file_path)
            )
        )

        # ---------------------------------------------------------
        # Step 5: Extract and store entities per chunk
        # ---------------------------------------------------------
        total_entities = 0
        total_relationships = 0
        processed_entity_chunks = 0

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            chunk_text = chunk["text"]

            extraction = self.entity_extractor.extract(
                chunk_text
            )

            if not extraction.get("entities") and not extraction.get(
                "relationships"
            ):
                continue

            extraction_result = (
                self.entity_graph.store_extraction(
                    chunk_id=chunk_id,
                    extraction=extraction,
                )
            )

            total_entities += extraction_result[
                "entities_stored"
            ]

            total_relationships += extraction_result[
                "relationships_stored"
            ]

            processed_entity_chunks += 1

        return {
            "status": "success",
            "document_id": document_id,
            "filename": filename,
            "processed_file": str(processed_file_path),
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "embedding_dimension": (
                int(embeddings.shape[1])
                if embeddings.ndim == 2
                else 0
            ),
            "vector_index": self.vector_index_name,
            "document_graph": document_graph_result,
            "entity_chunks_processed": processed_entity_chunks,
            "entities_stored": total_entities,
            "relationships_stored": total_relationships,
        }