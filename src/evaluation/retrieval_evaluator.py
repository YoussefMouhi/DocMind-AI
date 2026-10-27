"""
DocMind AI — Real FAISS Retrieval Evaluation
"""

import json
import time
from pathlib import Path

from src.embeddings.embedder import DocumentEmbedder
from src.storage.persistence import load_session
from src.evaluation.retrieval_metrics import evaluate_retrieval


class RetrievalEvaluator:

    def __init__(self):
        session = None

        from src.retrieval.vector_store import VectorStore
        import json
        from pathlib import Path

        index_dir = Path("data/evaluation/vectorstore")
        chunks_file = Path("data/evaluation/indexed_chunks.json")

        store = VectorStore.load(index_dir)

        chunks = json.loads(
            chunks_file.read_text(encoding="utf-8")
        )

        session = {
            "vector_store": store,
            "chunks": chunks
        }

        if session is None:
            raise RuntimeError(
                "No saved FAISS index found. "
                "Process documents in DocMind AI first."
            )

        self.store = session["vector_store"]
        self.chunks = session["chunks"]
        self.embedder = DocumentEmbedder()

    def search(self, question, k=3):
        start = time.perf_counter()

        embedding = self.embedder.encode([question])

        results = self.store.search(embedding, top_k=k)

        latency_ms = (time.perf_counter() - start) * 1000

        return results, latency_ms

    def evaluate_question(self, question, relevant_ids, k=3):
        results, latency_ms = self.search(question, k=k)

        retrieved_ids = [
            result["chunk_id"]
            for result in results
        ]

        metrics = evaluate_retrieval(
            retrieved_ids=retrieved_ids,
            relevant_ids=relevant_ids,
            k=k,
        )

        return {
            "question": question,
            "retrieved_ids": retrieved_ids,
            "relevant_ids": relevant_ids,
            "precision_at_k": metrics["precision_at_k"],
            "recall_at_k": metrics["recall_at_k"],
            "reciprocal_rank": metrics["reciprocal_rank"],
            "latency_ms": round(latency_ms, 2),
        }

    def evaluate_dataset(self, dataset, k=3):
        results = []

        for item in dataset:
            result = self.evaluate_question(
                question=item["question"],
                relevant_ids=item["relevant_ids"],
                k=k,
            )

            results.append(result)

        if not results:
            raise ValueError("Evaluation dataset is empty.")

        summary = {
            "total_questions": len(results),
            "precision_at_k": sum(
                item["precision_at_k"] for item in results
            ) / len(results),
            "recall_at_k": sum(
                item["recall_at_k"] for item in results
            ) / len(results),
            "mrr": sum(
                item["reciprocal_rank"] for item in results
            ) / len(results),
            "avg_latency_ms": sum(
                item["latency_ms"] for item in results
            ) / len(results),
        }

        return {
            "summary": summary,
            "results": results,
        }


def save_evaluation(report, path="data/evaluation/results.json"):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    return output
