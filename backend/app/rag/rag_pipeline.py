"""
End-to-end Hybrid RAG Pipeline.

Flow:

User Query
    ↓
Hybrid Retrieval
    ↓
Context Builder
    ↓
Answer Generator
    ↓
Final Grounded Response
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from app.retrieval.hybrid_retriever import HybridRetriever
from app.rag.context_builder import ContextBuilder
from app.rag.answer_generator import AnswerGenerator


class RAGPipeline:
    """
    Orchestrates retrieval, context construction, and answer generation.
    """

    def __init__(
        self,
        retriever: Optional[Any] = None,
        context_builder: Optional[ContextBuilder] = None,
        answer_generator: Optional[AnswerGenerator] = None,
    ):
        self.retriever = retriever or HybridRetriever()
        self.context_builder = (
            context_builder or ContextBuilder()
        )
        self.answer_generator = (
            answer_generator or AnswerGenerator()
        )

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_value(
        data: Dict[str, Any],
        keys: list[str],
        default: Any = None,
    ) -> Any:
        """
        Return the first available value from a dictionary.
        """
        for key in keys:
            if key in data and data[key] is not None:
                return data[key]

        return default

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        """
        Normalize a value to a list.
        """
        if value is None:
            return []

        if isinstance(value, list):
            return value

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, dict):
            return [value]

        return [value]

    def _get_sources(
        self,
        hybrid_result: Dict[str, Any],
    ) -> list[Dict[str, Any]]:
        """
        Extract source metadata from retrieved chunks.
        """
        chunks = self._get_value(
            hybrid_result,
            [
                "combined_chunks",
                "chunks",
                "documents",
            ],
            default=[],
        )

        chunks = self._as_list(chunks)

        sources = []

        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue

            sources.append(
                {
                    "chunk_id": self._get_value(
                        chunk,
                        ["chunk_id", "id"],
                    ),
                    "document_id": self._get_value(
                        chunk,
                        ["document_id", "doc_id"],
                    ),
                    "page_number": self._get_value(
                        chunk,
                        ["page_number", "page"],
                    ),
                    "source_types": self._get_value(
                        chunk,
                        [
                            "source_types",
                            "sources",
                            "retrieved_by",
                        ],
                        default=[],
                    ),
                    "score": self._get_value(
                        chunk,
                        [
                            "score",
                            "vector_score",
                            "similarity",
                        ],
                    ),
                }
            )

        return sources

    def _get_retrieval_summary(
        self,
        hybrid_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Return retrieval summary, calculating missing values when needed.
        """
        existing_summary = self._get_value(
            hybrid_result,
            [
                "summary",
                "retrieval_summary",
            ],
            default={},
        )

        if not isinstance(existing_summary, dict):
            existing_summary = {}

        vector_results = self._get_value(
            hybrid_result,
            [
                "vector_results",
                "vector_chunks",
            ],
            default=[],
        )

        graph_entities = self._get_value(
            hybrid_result,
            [
                "graph_entities",
                "entities",
            ],
            default=[],
        )

        graph_relationships = self._get_value(
            hybrid_result,
            [
                "graph_relationships",
                "relationships",
                "relations",
            ],
            default=[],
        )

        graph_chunks = self._get_value(
            hybrid_result,
            [
                "graph_chunks",
            ],
            default=[],
        )

        combined_chunks = self._get_value(
            hybrid_result,
            [
                "combined_chunks",
                "chunks",
            ],
            default=[],
        )

        calculated_summary = {
            "vector_result_count": len(
                self._as_list(vector_results)
            ),
            "graph_entity_count": len(
                self._as_list(graph_entities)
            ),
            "graph_relationship_count": len(
                self._as_list(graph_relationships)
            ),
            "graph_chunk_count": len(
                self._as_list(graph_chunks)
            ),
            "combined_chunk_count": len(
                self._as_list(combined_chunks)
            ),
        }

        calculated_summary.update(existing_summary)

        return calculated_summary

    def _get_context_statistics(
        self,
        hybrid_result: Dict[str, Any],
        context: str,
    ) -> Dict[str, Any]:
        """
        Obtain context statistics using the updated ContextBuilder.
        """
        stats = self.context_builder.get_context_statistics(
            hybrid_result=hybrid_result,
            context=context,
        )

        if not isinstance(stats, dict):
            stats = {}

        return stats

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        query: str,
        max_context_chunks: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute the complete Hybrid RAG pipeline.
        """
        started_at = time.perf_counter()

        query = (query or "").strip()

        if not query:
            return {
                "success": False,
                "query": query,
                "error": "Query cannot be empty.",
            }

        try:
            # ----------------------------------------------------------
            # Step 1: Hybrid retrieval
            # ----------------------------------------------------------
            retrieval_started = time.perf_counter()

            if hasattr(self.retriever, "retrieve"):
                hybrid_result = self.retriever.retrieve(query)

            elif hasattr(self.retriever, "search"):
                hybrid_result = self.retriever.search(query)

            else:
                raise AttributeError(
                    "Retriever must implement retrieve() or search()."
                )

            retrieval_seconds = (
                time.perf_counter() - retrieval_started
            )

            if not isinstance(hybrid_result, dict):
                raise ValueError(
                    "Hybrid retriever must return a dictionary."
                )

            # ----------------------------------------------------------
            # Step 2: Build context
            # ----------------------------------------------------------
            context_started = time.perf_counter()

            context = self.context_builder.build_context(
                query=query,
                hybrid_results=hybrid_result,
                max_context_chunks=max_context_chunks,
            )

            context_seconds = (
                time.perf_counter() - context_started
            )

            # ----------------------------------------------------------
            # Step 3: Generate answer
            # ----------------------------------------------------------
            answer_started = time.perf_counter()

            answer_result = (
                self.answer_generator.generate_answer(
                    query=query,
                    context=context,
                )
            )

            answer_seconds = (
                time.perf_counter() - answer_started
            )

            if isinstance(answer_result, dict):
                answer = answer_result.get(
                    "answer",
                    "",
                )

                model = answer_result.get(
                    "model",
                )

                guardrail_status = answer_result.get(
                    "guardrail_status",
                    "unknown",
                )
            else:
                answer = str(answer_result)
                model = None
                guardrail_status = "unknown"

            retrieval_summary = (
                self._get_retrieval_summary(
                    hybrid_result
                )
            )

            sources = self._get_sources(
                hybrid_result
            )

            context_statistics = (
                self._get_context_statistics(
                    hybrid_result=hybrid_result,
                    context=context,
                )
            )

            total_seconds = (
                time.perf_counter() - started_at
            )

            return {
                "success": True,
                "query": query,
                "answer": answer,
                "sources": sources,
                "retrieval_summary": retrieval_summary,
                "context_statistics": context_statistics,
                "guardrail_status": guardrail_status,
                "model": model,
                "timings": {
                    "retrieval_seconds": round(
                        retrieval_seconds,
                        4,
                    ),
                    "context_building_seconds": round(
                        context_seconds,
                        4,
                    ),
                    "answer_generation_seconds": round(
                        answer_seconds,
                        4,
                    ),
                    "total_seconds": round(
                        total_seconds,
                        4,
                    ),
                },
                "debug": {
                    "graph_entities_available": (
                        retrieval_summary.get(
                            "graph_entity_count",
                            0,
                        )
                    ),
                    "graph_relationships_available": (
                        retrieval_summary.get(
                            "graph_relationship_count",
                            0,
                        )
                    ),
                    "graph_context_included": (
                        context_statistics.get(
                            "graph_context_included",
                            False,
                        )
                    ),
                    "vector_context_included": (
                        context_statistics.get(
                            "vector_context_included",
                            False,
                        )
                    ),
                },
                "error": None,
            }

        except Exception as exc:
            total_seconds = (
                time.perf_counter() - started_at
            )

            return {
                "success": False,
                "query": query,
                "answer": None,
                "sources": [],
                "retrieval_summary": {},
                "context_statistics": {},
                "guardrail_status": "failed",
                "model": None,
                "timings": {
                    "total_seconds": round(
                        total_seconds,
                        4,
                    ),
                },
                "error": str(exc),
            }