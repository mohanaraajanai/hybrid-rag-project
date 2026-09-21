import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]

VECTOR_STORE_DIRECTORY = (
    PROJECT_ROOT / "data" / "vector_store"
)

VECTOR_STORE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


class FAISSVectorStore:
    """
    Persistent FAISS vector store with JSON metadata.
    """

    def __init__(
        self,
        index_name: str = "documents",
    ):
        self.index_name = index_name

        self.index_path = (
            VECTOR_STORE_DIRECTORY / f"{index_name}.faiss"
        )

        self.metadata_path = (
            VECTOR_STORE_DIRECTORY / f"{index_name}_metadata.json"
        )

        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []

    def create_index(
        self,
        embeddings: np.ndarray,
    ) -> None:
        """
        Create a FAISS inner-product index.

        Embeddings should already be normalized.
        """

        if embeddings.size == 0:
            raise ValueError(
                "Cannot create an index without embeddings."
            )

        embedding_dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            embedding_dimension
        )

        self.index.add(embeddings)

    def add_metadata(
        self,
        metadata: list[dict[str, Any]],
    ) -> None:
        """
        Store metadata in the same order as FAISS vectors.
        """

        self.metadata.extend(metadata)

    def save(self) -> None:
        """
        Persist FAISS index and metadata to disk.
        """

        if self.index is None:
            raise ValueError(
                "FAISS index has not been created."
            )

        if self.index.ntotal != len(self.metadata):
            raise ValueError(
                "Number of vectors and metadata records do not match."
            )

        faiss.write_index(
            self.index,
            str(self.index_path),
        )

        self.metadata_path.write_text(
            json.dumps(
                self.metadata,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self) -> None:
        """
        Load an existing FAISS index and metadata.
        """

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {self.index_path}"
            )

        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {self.metadata_path}"
            )

        self.index = faiss.read_index(
            str(self.index_path)
        )

        self.metadata = json.loads(
            self.metadata_path.read_text(
                encoding="utf-8"
            )
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Search for the most similar chunks.
        """

        if self.index is None:
            raise ValueError(
                "FAISS index has not been loaded or created."
            )

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        actual_top_k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding.astype("float32"),
            actual_top_k,
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index == -1:
                continue

            result = {
                "score": float(score),
                "metadata": self.metadata[index],
            }

            results.append(result)

        return results