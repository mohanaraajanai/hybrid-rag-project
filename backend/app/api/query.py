from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.rag.rag_pipeline import RAGPipeline


router = APIRouter(
    prefix="/api/query",
    tags=["RAG Query"],
)


class QueryRequest(BaseModel):
    """
    Request model for Hybrid RAG queries.
    """

    query: str = Field(
        ...,
        min_length=1,
        description="Question to ask the Hybrid RAG system.",
    )

    max_context_chunks: Optional[int] = Field(
        default=None,
        ge=1,
        description="Optional maximum number of chunks used for context.",
    )


@router.post("")
async def query_rag(request: QueryRequest):
    """
    Execute the complete Hybrid RAG pipeline.
    """

    try:
        pipeline = RAGPipeline()

        result = pipeline.run(
            query=request.query,
            max_context_chunks=request.max_context_chunks,
        )

        if not result.get("success", False):
            return {
                "success": False,
                "query": request.query,
                "answer": None,
                "sources": [],
                "retrieval_summary": result.get(
                    "retrieval_summary",
                    {},
                ),
                "context_statistics": result.get(
                    "context_statistics",
                    {},
                ),
                "guardrail_status": result.get(
                    "guardrail_status",
                    "failed",
                ),
                "model": result.get("model"),
                "timings": result.get(
                    "timings",
                    {},
                ),
                "debug": result.get(
                    "debug",
                    {},
                ),
                "error": result.get(
                    "error",
                    "RAG query failed.",
                ),
            }

        return result

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {exc}",
        ) from exc