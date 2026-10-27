import json
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.quality_evaluator import RAGQualityEvaluator


INPUT = Path("data/evaluation/answer_results.json")
OUTPUT = Path("data/evaluation/quality_results.json")


def main():
    if not INPUT.exists():
        raise SystemExit(
            "Missing answer evaluation results. "
            "Run python evaluate_answers.py first."
        )

    data = json.loads(INPUT.read_text(encoding="utf-8"))
    results = data.get("results", [])

    if not results:
        raise SystemExit("No evaluation cases found.")

    evaluator = RAGQualityEvaluator()

    evaluated = []

    print("\nDOCMIND AI — RAG QUALITY EVALUATION")
    print("=" * 60)

    for index, item in enumerate(results, start=1):
        question = item["question"]
        answer = item["answer"]
        sources = item.get("sources", [])

        print(f"\nEvaluating {index}/{len(results)}")
        print(f"Question: {question}")

        try:
            scores = evaluator.evaluate(
                question=question,
                answer=answer,
                contexts=sources,
            )
        except Exception as exc:
            print(f"Evaluation failed: {exc}")
            evaluated.append({
                **item,
                "quality_evaluation": {
                    "status": "error",
                    "error": str(exc),
                },
            })
            continue

        evaluated.append({
            **item,
            "quality_evaluation": {
                "status": "success",
                **scores,
            },
        })

        print(f"Groundedness: {scores['groundedness']}/5")
        print(f"Relevance: {scores['relevance']}/5")
        print(f"Explanation: {scores['explanation']}")

    successful = [
        item["quality_evaluation"]
        for item in evaluated
        if item["quality_evaluation"]["status"] == "success"
    ]

    summary = {
        "total_cases": len(evaluated),
        "evaluated_cases": len(successful),
        "failed_cases": len(evaluated) - len(successful),
        "average_groundedness": (
            round(
                sum(x["groundedness"] for x in successful)
                / len(successful),
                3,
            )
            if successful else None
        ),
        "average_relevance": (
            round(
                sum(x["relevance"] for x in successful)
                / len(successful),
                3,
            )
            if successful else None
        ),
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator_model": data.get("model"),
        "judge_model": evaluator.judge_model,
        "summary": summary,
        "results": evaluated,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nResults saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
