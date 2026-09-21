from datetime import datetime, timezone

from fastapi import FastAPI

from app.api.upload import router as upload_router
from app.api.query import router as query_router


app = FastAPI(
    title="Hybrid RAG API",
    version="1.0.0",
    description="Hybrid RAG document processing backend",
)


app.include_router(upload_router)
app.include_router(query_router)


@app.get("/")
def root():
    return {
        "message": "Hybrid RAG API is running",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }