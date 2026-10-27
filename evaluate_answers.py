import json
import re
from pathlib import Path

from src.generation.rag_engine import RAGEngine

CASES = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "expected_behavior": "answer",
        "chunks": [
            {
                "text": (
                    "Retrieval-Augmented Generation (RAG) combines "
                    "information retrieval with language model generation. "
                    "A retriever finds relevant document passages, and "
                    "a language model uses them to generate grounded answers."
                ),
                "source": "rag_reference.txt",
                "page": None,
                "similarity_score": 0.95,
            }
        ],
    },
    {
        "question": "What is FAISS used for?",
        "expected_behavior": "answer",
        "chunks": [
            {
                "text": (
                    "FAISS is a library for efficient similarity search "
                    "over dense vectors. It can retrieve vectors nearest "
                    "to a query embedding."
                ),
                "source": "faiss_reference.txt",
                "page": None,
                "similarity_score": 0.93,
            }
        ],
    },
    {
        "question": "What is the capital of Mars?",
        "expected_behavior": "abstain",
        "chunks": [
            {
                "text": (
                    "FAISS is a library for efficient similarity search "
                    "over dense vectors."
                ),
                "source": "faiss_reference.txt",
                "page": None,
                "similarity_score": 0.20,
            }
        ],
    },
    {
        "question": "What does the document say about quantum teleportation?",
        "expected_behavior": "abstain",
        "chunks": [],
    },
]

def extract_citations(answer):
    return [int(value) for value in re.findall(r"\[(\d+)\]", answer)]

def evaluate_case(engine, case):
    result = engine.generate_answer(
        question=case["question"],
        retrieved_chunks=case["chunks"],
    )

    answer = result["answer"]
    sources = result["sources"]

    citations = extract_citations(answer)
    valid_citations = all(1 <= citation <= len(sources) for citation in citations)

    return {
        "question": case["question"],
        "expected_behavior": case["expected_behavior"],
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "citations_valid": valid_citations,
        "has_citation": bool(citations),
        "manual_review": {
            "answer_relevant": None,
            "factually_grounded": None,
            "correct_abstention": None,
        },
    }

def main():
    engine = RAGEngine()

    results = []

    for index, case in enumerate(CASES, start=1):
        print(f"\nEvaluating question {index}/{len(CASES)}")
        print(case["question"])

        result = evaluate_case(engine, case)
        results.append(result)

        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCitations valid: {result['citations_valid']}")
        print(f"Citations found: {result['citations']}")

    report = {
        "model": "qwen2.5:3b",
        "total_questions": len(results),
        "results": results,
    }

    output = Path("data/evaluation/answer_results.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nEvaluation saved to: {output}")
    print("Manual review is required to assess factual grounding.")

if __name__ == "__main__":
    main()
