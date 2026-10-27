import json

from src.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
    save_evaluation,
)

with open("data/evaluation/questions.json", encoding="utf-8") as f:
    dataset = json.load(f)

print("\nInitializing DocMind AI Retrieval Evaluator...")

evaluator = RetrievalEvaluator()

print(f"Loaded {len(evaluator.chunks)} indexed chunks.")

print("\nRunning retrieval evaluation...\n")

report = evaluator.evaluate_dataset(dataset, k=3)

for result in report["results"]:
    print("=" * 60)
    print("QUESTION:", result["question"])
    print("Retrieved IDs:", result["retrieved_ids"])
    print("Relevant IDs:", result["relevant_ids"])
    print(f"Precision@3: {result['precision_at_k']:.4f}")
    print(f"Recall@3: {result['recall_at_k']:.4f}")
    print(f"Reciprocal Rank: {result['reciprocal_rank']:.4f}")
    print(f"Latency: {result['latency_ms']:.2f} ms")

summary = report["summary"]

print("\n" + "=" * 60)
print("DOCMIND AI — RETRIEVAL EVALUATION SUMMARY")
print("=" * 60)

print(f"Questions evaluated: {summary['total_questions']}")
print(f"Mean Precision@3: {summary['precision_at_k']:.4f}")
print(f"Mean Recall@3: {summary['recall_at_k']:.4f}")
print(f"MRR: {summary['mrr']:.4f}")
print(f"Average latency: {summary['avg_latency_ms']:.2f} ms")

output = save_evaluation(report)

print(f"\nEvaluation saved to: {output}")
print("\nEvaluation completed successfully!")
