import json
from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add_documents(self, chunks, embeddings):
        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")

        if len(chunks) != len(vectors):
            raise ValueError("Chunks and embeddings must match")

        if vectors.shape[1] != self.index.d:
            raise ValueError("Embedding dimension mismatch")

        self.index.add(vectors)
        self.documents.extend(chunks)

    def search(self, query_embedding, top_k=3):
        if top_k <= 0 or not self.documents:
            return []

        query = np.asarray(
            query_embedding, dtype=np.float32
        ).reshape(1, -1)

        if query.shape[1] != self.index.d:
            raise ValueError("Query dimension mismatch")

        k = min(top_k, len(self.documents))
        scores, indices = self.index.search(query, k)

        return [
            {
                **self.documents[int(index)],
                "similarity_score": float(score),
            }
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss")
        )

        with open(
            directory / "metadata.json",
            "w",
            encoding="utf-8"
        ) as file:
            json.dump(
                self.documents,
                file,
                ensure_ascii=False,
                indent=2
            )

    @classmethod
    def load(cls, directory):
        directory = Path(directory)

        index_path = directory / "index.faiss"
        metadata_path = directory / "metadata.json"

        if not index_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(
                "Saved FAISS index or metadata not found"
            )

        # Only load indexes created locally by this application.
        # Never load an untrusted FAISS index file.
        index = faiss.read_index(str(index_path))

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:
            documents = json.load(file)

        if index.ntotal != len(documents):
            raise ValueError(
                "FAISS index and metadata are inconsistent"
            )

        store = cls(index.d)
        store.index = index
        store.documents = documents

        return store
