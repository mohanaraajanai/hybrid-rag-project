from typing import Any

from app.retrieval.vector_retriever import VectorRetriever
from app.retrieval.graph_retriever import GraphRetriever


class HybridRetriever:
    """
    Combines FAISS vector retrieval and Neo4j graph retrieval.

    Vector and graph storage now use the same chunk IDs, so chunk_id
    is the primary identity used for deduplication.

    The retriever also normalizes source metadata so the API can expose
    clean citation/source information.
    """

    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()

    def close(self):
        """Close graph database resources."""
        self.graph_retriever.close()

    @staticmethod
    def _get_deduplication_key(
        chunk: dict[str, Any],
    ) -> str:
        """
        Generate a stable deduplication key.

        Primary:
            chunk_id

        Fallback:
            document_id + page_number + page_chunk_index

        Final fallback:
            document_id + page_number

        Important:
            We do NOT use document_id + page_number as the primary key
            because a single PDF page may contain multiple chunks.
        """

        chunk_id = chunk.get("chunk_id")

        if chunk_id:
            return f"chunk::{chunk_id}"

        document_id = chunk.get("document_id")
        page_number = chunk.get("page_number")
        page_chunk_index = chunk.get("page_chunk_index")

        if (
            document_id
            and page_number is not None
            and page_chunk_index is not None
        ):
            return (
                f"location::{document_id}"
                f"::{page_number}"
                f"::{page_chunk_index}"
            )

        if document_id and page_number is not None:
            return (
                f"page::{document_id}::{page_number}"
            )

        return "unknown-chunk"

    @staticmethod
    def _merge_sources(
        existing_sources: list[str],
        new_sources: list[str],
    ) -> list[str]:
        """
        Merge retrieval source labels without duplicates.
        """

        merged = list(existing_sources or [])

        for source in new_sources:
            if source not in merged:
                merged.append(source)

        return merged

    @staticmethod
    def _derive_document_id_from_chunk_id(
        chunk_id: str | None,
    ) -> str | None:
        """
        Recover document_id from the current chunk ID format.

        Current chunk IDs are:

            {document_id}_{chunk_index}

        Example:

            c32f1170-ed7a-4e97-b0f4-dbf2317c0c81_5
        """

        if not chunk_id:
            return None

        if "_" not in chunk_id:
            return None

        candidate, chunk_index = chunk_id.rsplit(
            "_",
            1,
        )

        if not chunk_index.isdigit():
            return None

        return candidate

    @staticmethod
    def _merge_metadata(
        target: dict[str, Any],
        fallback: dict[str, Any],
    ) -> None:
        """
        Fill missing metadata in target using fallback metadata.
        """

        for field in (
            "document_id",
            "filename",
            "page_number",
            "page_chunk_index",
            "text",
        ):
            current_value = target.get(field)

            if (
                current_value is None
                or current_value == ""
            ):
                fallback_value = fallback.get(field)

                if (
                    fallback_value is not None
                    and fallback_value != ""
                ):
                    target[field] = fallback_value

    def retrieve(
        self,
        query: str,
        vector_top_k: int = 5,
        graph_max_entities: int = 15,
        graph_max_chunks: int = 10,
    ) -> dict[str, Any]:
        """
        Retrieve and combine results from FAISS and Neo4j.
        """

        if not query or not query.strip():
            raise ValueError(
                "Hybrid retrieval query cannot be empty."
            )

        # =========================================================
        # 1. Vector retrieval
        # =========================================================

        vector_results = self.vector_retriever.retrieve(
            query=query,
            top_k=vector_top_k,
        )

        # =========================================================
        # 2. Build vector metadata lookup
        # =========================================================
        #
        # This lets graph chunks inherit missing metadata when the
        # same chunk_id also exists in the vector results.
        # =========================================================

        vector_metadata_by_chunk_id: dict[
            str,
            dict[str, Any],
        ] = {}

        for result in vector_results:
            metadata = result.get(
                "metadata",
                {},
            )

            chunk_id = metadata.get(
                "chunk_id"
            )

            if not chunk_id:
                continue

            vector_metadata_by_chunk_id[
                chunk_id
            ] = metadata

        # =========================================================
        # 3. Graph retrieval
        # =========================================================

        graph_results = self.graph_retriever.retrieve(
            query=query,
            max_entities=graph_max_entities,
            max_chunks=graph_max_chunks,
        )

        # =========================================================
        # 4. Combine and deduplicate
        # =========================================================

        combined_chunks: dict[
            str,
            dict[str, Any],
        ] = {}

        # =========================================================
        # 4A. Process vector results
        # =========================================================

        for result in vector_results:
            metadata = result.get(
                "metadata",
                {},
            )

            chunk = {
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "document_id": metadata.get(
                    "document_id"
                ),
                "filename": metadata.get(
                    "filename"
                ),
                "page_number": metadata.get(
                    "page_number"
                ),
                "page_chunk_index": metadata.get(
                    "page_chunk_index"
                ),
                "text": metadata.get(
                    "text",
                    "",
                ),
                "vector_score": result.get(
                    "score"
                ),
                "retrieval_sources": [
                    "vector"
                ],
                "source_types": [
                    "vector"
                ],
            }

            if not chunk["chunk_id"]:
                continue

            dedup_key = self._get_deduplication_key(
                chunk
            )

            if dedup_key not in combined_chunks:
                combined_chunks[
                    dedup_key
                ] = chunk

            else:
                existing = combined_chunks[
                    dedup_key
                ]

                merged_sources = (
                    self._merge_sources(
                        existing.get(
                            "retrieval_sources",
                            [],
                        ),
                        ["vector"],
                    )
                )

                existing[
                    "retrieval_sources"
                ] = merged_sources

                existing[
                    "source_types"
                ] = list(merged_sources)

                current_score = chunk.get(
                    "vector_score"
                )

                existing_score = existing.get(
                    "vector_score"
                )

                if (
                    current_score is not None
                    and (
                        existing_score is None
                        or current_score
                        > existing_score
                    )
                ):
                    existing[
                        "vector_score"
                    ] = current_score

        # =========================================================
        # 4B. Process graph chunks
        # =========================================================

        for result in graph_results.get(
            "chunks",
            [],
        ):
            chunk_id = result.get(
                "chunk_id"
            )

            vector_metadata = (
                vector_metadata_by_chunk_id.get(
                    chunk_id,
                    {},
                )
            )

            chunk = {
                "chunk_id": chunk_id,
                "document_id": result.get(
                    "document_id"
                ),
                "filename": result.get(
                    "filename"
                ),
                "page_number": result.get(
                    "page_number"
                ),
                "page_chunk_index": result.get(
                    "page_chunk_index"
                ),
                "text": result.get(
                    "text",
                    "",
                ),
                "vector_score": None,
                "retrieval_sources": [
                    "graph"
                ],
                "source_types": [
                    "graph"
                ],
            }

            if not chunk["chunk_id"]:
                continue

            # -----------------------------------------------------
            # Fill missing graph metadata from vector metadata
            # when both stores know the same chunk.
            # -----------------------------------------------------

            self._merge_metadata(
                target=chunk,
                fallback=vector_metadata,
            )

            # -----------------------------------------------------
            # Current chunk IDs contain document_id, so recover it
            # when the graph query did not return it.
            # -----------------------------------------------------

            if not chunk.get("document_id"):
                chunk[
                    "document_id"
                ] = self._derive_document_id_from_chunk_id(
                    chunk_id
                )

            dedup_key = self._get_deduplication_key(
                chunk
            )

            if dedup_key in combined_chunks:
                existing = combined_chunks[
                    dedup_key
                ]

                merged_sources = (
                    self._merge_sources(
                        existing.get(
                            "retrieval_sources",
                            [],
                        ),
                        ["graph"],
                    )
                )

                existing[
                    "retrieval_sources"
                ] = merged_sources

                existing[
                    "source_types"
                ] = list(merged_sources)

                # Fill any missing metadata.
                self._merge_metadata(
                    target=existing,
                    fallback=chunk,
                )

            else:
                combined_chunks[
                    dedup_key
                ] = chunk

        # =========================================================
        # 5. Sort combined chunks
        # =========================================================

        combined_chunk_list = list(
            combined_chunks.values()
        )

        combined_chunk_list.sort(
            key=lambda item: (
                item.get("vector_score") is not None,
                item.get("vector_score") or 0,
            ),
            reverse=True,
        )

        # =========================================================
        # 6. Prepare unified result
        # =========================================================

        return {
            "query": query,
            "vector_results": vector_results,
            "graph_results": graph_results,
            "combined_chunks": combined_chunk_list,
            "summary": {
                "vector_result_count": len(
                    vector_results
                ),
                "graph_entity_count": len(
                    graph_results.get(
                        "entities",
                        [],
                    )
                ),
                "graph_relationship_count": len(
                    graph_results.get(
                        "relationships",
                        [],
                    )
                ),
                "graph_chunk_count": len(
                    graph_results.get(
                        "chunks",
                        [],
                    )
                ),
                "combined_chunk_count": len(
                    combined_chunk_list
                ),
            },
        }