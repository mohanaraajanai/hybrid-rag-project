from typing import Any

from app.embeddings.local_embeddings import generate_query_embedding
from app.vector_store.faiss_store import FAISSVectorStore


class VectorRetriever:
    """
    Retrieve relevant document chunks from the FAISS vector store.
    """

    def __init__(
        self,
        index_name: str = "documents",
    ):
        self.vector_store = FAISSVectorStore(
            index_name=index_name
        )

        self.vector_store.load()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a user query.

        Args:
            query: User's search question.
            top_k: Maximum number of results to return.

        Returns:
            List of retrieved chunks with scores and metadata.
        """

        if not query or not query.strip():
            raise ValueError(
                "Retrieval query cannot be empty."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # Generate embedding for the user question.
        query_embedding = generate_query_embedding(query)

        # Search the FAISS index.
        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        return results