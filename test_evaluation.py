from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    evaluate_retrieval,
)


def test_precision():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [1, 3, 5]

    result = precision_at_k(retrieved, relevant, k=5)

    assert result == 0.6


def test_recall():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [1, 3, 5, 7]

    result = recall_at_k(retrieved, relevant, k=5)

    assert result == 0.75


def test_reciprocal_rank():
    retrieved = [4, 2, 3, 1]
    relevant = [3, 1]

    result = reciprocal_rank(retrieved, relevant)

    assert abs(result - 1 / 3) < 1e-9


def test_complete_evaluation():
    retrieved = [1, 2, 3, 4, 5]
    relevant = [1, 3, 5]

    results = evaluate_retrieval(
        retrieved_ids=retrieved,
        relevant_ids=relevant,
        k=5,
    )

    assert results["precision_at_k"] == 0.6
    assert results["recall_at_k"] == 1.0
    assert results["reciprocal_rank"] == 1.0
