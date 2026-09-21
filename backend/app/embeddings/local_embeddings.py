from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def generate_embeddings(
    texts: list[str],
) -> np.ndarray:
    """
    Generate normalized embeddings for a list of texts.

    Returns:
        NumPy array with shape:
        (number_of_texts, embedding_dimension)
    """

    if not texts:
        return np.empty((0, 384), dtype="float32")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return embeddings.astype("float32")


def generate_query_embedding(
    query: str,
) -> np.ndarray:
    """
    Generate a normalized embedding for a search query.
    """

    if not query or not query.strip():
        raise ValueError("Search query cannot be empty.")

    return generate_embeddings([query])