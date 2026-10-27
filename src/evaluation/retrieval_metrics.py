"""
DocMind AI — Information Retrieval Evaluation Metrics.

Metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
"""


def precision_at_k(retrieved_ids, relevant_ids, k=5):
    """Fraction of the top K results that are relevant."""
    if k <= 0:
        raise ValueError("k must be positive")

    retrieved = retrieved_ids[:k]
    relevant = set(relevant_ids)

    hits = sum(doc_id in relevant for doc_id in retrieved)

    return hits / k


def recall_at_k(retrieved_ids, relevant_ids, k=5):
    """Fraction of all relevant documents retrieved in the top K."""
    relevant = set(relevant_ids)

    if not relevant:
        return 0.0

    retrieved = set(retrieved_ids[:k])

    return len(retrieved & relevant) / len(relevant)


def reciprocal_rank(retrieved_ids, relevant_ids):
    """Reciprocal rank of the first relevant result."""
    relevant = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(retrieved_ids, relevant_ids, k=5):
    """Calculate retrieval evaluation metrics."""
    return {
        "precision_at_k": precision_at_k(
            retrieved_ids, relevant_ids, k
        ),
        "recall_at_k": recall_at_k(
            retrieved_ids, relevant_ids, k
        ),
        "reciprocal_rank": reciprocal_rank(
            retrieved_ids, relevant_ids
        ),
    }
