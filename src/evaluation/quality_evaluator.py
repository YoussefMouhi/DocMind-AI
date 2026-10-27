import json
import re

from ollama import Client


class RAGQualityEvaluator:
    def __init__(
        self,
        judge_model="llama3.1:8b",
        host="http://localhost:11434",
    ):
        self.client = Client(host=host)
        self.judge_model = judge_model

    def evaluate(self, question, answer, contexts):
        context_text = "\n\n".join(
            context.get("text", "")
            if isinstance(context, dict)
            else str(context)
            for context in contexts
        )

        prompt = f"""
You are evaluating a retrieval-augmented generation system.

Evaluate the answer using only the question and document excerpts.

SCORING:
- groundedness: 0 to 5
  Does the provided evidence support the answer?
- relevance: 0 to 5
  Does the answer address the question?

If the answer appropriately states that information is unavailable,
do not penalize it for refusing to invent facts.

Return ONLY valid JSON with this structure:
{{
  "groundedness": 0,
  "relevance": 0,
  "explanation": "Brief justification"
}}

QUESTION:
{question}

DOCUMENT EXCERPTS:
{context_text}

GENERATED ANSWER:
{answer}
"""

        response = self.client.chat(
            model=self.judge_model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            format="json",
            options={"temperature": 0},
        )

        content = response["message"]["content"]
        result = json.loads(content)

        scores = {}

        for metric in ("groundedness", "relevance"):
            value = float(result[metric])

            if not 0 <= value <= 5:
                raise ValueError(
                    f"{metric} must be between 0 and 5."
                )

            scores[metric] = value

        return {
            "groundedness": scores["groundedness"],
            "relevance": scores["relevance"],
            "explanation": str(result.get("explanation", "")),
            "judge_model": self.judge_model,
        }
