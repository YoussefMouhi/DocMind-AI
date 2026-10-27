import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


REPORT_PATH = Path("data/evaluation/results.json")


def render_evaluation_page():
    st.title("📊 Model Evaluation")

    st.caption(
        "Retrieval performance analysis | "
        "Precision@K • Recall@K • MRR • Latency"
    )

    if not REPORT_PATH.exists():
        st.warning(
            "No evaluation report found. "
            "Run python run_evaluation.py first."
        )
        return

    with REPORT_PATH.open(encoding="utf-8") as file:
        report = json.load(file)

    summary = report["summary"]
    results = report["results"]

    st.subheader("Retrieval Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Precision@3",
        f"{summary['precision_at_k']:.2%}"
    )

    col2.metric(
        "Recall@3",
        f"{summary['recall_at_k']:.2%}"
    )

    col3.metric(
        "MRR",
        f"{summary['mrr']:.3f}"
    )

    col4.metric(
        "Average Latency",
        f"{summary['avg_latency_ms']:.1f} ms"
    )

    st.info(
        f"Preliminary results from "
        f"{summary['total_questions']} evaluation questions. "
        "Relevance labels should be independently verified."
    )

    st.divider()

    st.subheader("Question-Level Performance")

    df = pd.DataFrame(results)

    display_df = df[
        [
            "question",
            "precision_at_k",
            "recall_at_k",
            "reciprocal_rank",
            "latency_ms",
        ]
    ].copy()

    display_df.columns = [
        "Question",
        "Precision@3",
        "Recall@3",
        "Reciprocal Rank",
        "Latency (ms)",
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    st.subheader("Retrieval Quality by Question")

    chart_df = df[
        ["question", "precision_at_k", "recall_at_k"]
    ].melt(
        id_vars="question",
        var_name="Metric",
        value_name="Score",
    )

    fig = px.bar(
        chart_df,
        x="question",
        y="Score",
        color="Metric",
        barmode="group",
        title="Precision and Recall Across Evaluation Questions",
        labels={"question": "Question"},
    )

    fig.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-25,
        yaxis_range=[0, 1.05],
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Retrieval Latency")

    latency_fig = px.bar(
        df,
        x="question",
        y="latency_ms",
        title="Retrieval Latency by Question",
        labels={
            "question": "Question",
            "latency_ms": "Latency (ms)",
        },
    )

    latency_fig.update_layout(
        template="plotly_dark",
        xaxis_tickangle=-25,
    )

    st.plotly_chart(latency_fig, use_container_width=True)

    st.divider()

    st.subheader("Retrieved Chunk Analysis")

    for result in results:
        with st.expander(result["question"]):
            st.write(
                "**Retrieved chunk IDs:**",
                result["retrieved_ids"],
            )
            st.write(
                "**Relevant chunk IDs:**",
                result["relevant_ids"],
            )
            st.write(
                "**Precision:**",
                round(result["precision_at_k"], 4),
            )
            st.write(
                "**Recall:**",
                round(result["recall_at_k"], 4),
            )

    st.caption(
        "Evaluation scope: retrieval only. "
        "Answer correctness and hallucination rates "
        "are not measured by this report."
    )

    render_answer_evaluation()



def render_quality_evaluation():
    """Display LLM-judged RAG answer quality metrics."""
    import json
    from pathlib import Path

    import pandas as pd
    import plotly.express as px
    import streamlit as st

    st.divider()
    st.header("🧠 RAG Quality Evaluation")
    st.caption("LLM-as-a-Judge | Groundedness | Answer Relevance")

    results_path = Path("data/evaluation/quality_results.json")

    if not results_path.exists():
        st.info(
            "No quality evaluation results found. "
            "Run the quality evaluation script first."
        )
        return

    data = json.loads(results_path.read_text(encoding="utf-8"))
    summary = data.get("summary", {})
    results = data.get("results", [])

    if not results:
        st.warning("The quality evaluation report contains no results.")
        return

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Evaluation Cases", summary.get("total_cases", 0))
    col2.metric("Evaluated", summary.get("evaluated_cases", 0))
    col3.metric(
        "Avg. Groundedness",
        f"{summary.get('average_groundedness', 0):.2f}/5",
    )
    col4.metric(
        "Avg. Relevance",
        f"{summary.get('average_relevance', 0):.2f}/5",
    )

    st.info(
        "These scores are generated by an LLM judge on a small "
        "evaluation set. They are not independently verified "
        "accuracy or hallucination-rate measurements."
    )

    rows = []

    for result in results:
        quality = result.get("quality_evaluation", {})

        if quality.get("status") != "success":
            continue

        rows.append({
            "Question": result.get("question", "Unknown"),
            "Groundedness": quality.get("groundedness"),
            "Relevance": quality.get("relevance"),
        })

    if rows:
        df = pd.DataFrame(rows)

        st.subheader("Quality Scores by Question")

        chart_df = df.melt(
            id_vars="Question",
            value_vars=["Groundedness", "Relevance"],
            var_name="Metric",
            value_name="Score",
        )

        fig = px.bar(
            chart_df,
            x="Question",
            y="Score",
            color="Metric",
            barmode="group",
            title="Groundedness and Relevance Across Questions",
        )

        fig.update_layout(
            template="plotly_dark",
            yaxis_range=[0, 5.5],
            xaxis_tickangle=-25,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Question-Level Scores")
        st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Detailed Quality Assessment")

    for index, result in enumerate(results, start=1):
        question = result.get("question", f"Question {index}")
        quality = result.get("quality_evaluation", {})

        with st.expander(f"{index}. {question}"):
            st.markdown("**Generated Answer**")
            st.write(result.get("answer", "Not available"))

            st.markdown("**Quality Evaluation**")
            st.write({
                "Status": quality.get("status", "unknown"),
                "Groundedness": quality.get("groundedness"),
                "Relevance": quality.get("relevance"),
                "Judge": quality.get("judge_model"),
            })

            st.markdown("**Judge Explanation**")
            st.write(quality.get("explanation", "No explanation available."))

    st.caption(
        "Evaluation scope: four controlled questions. "
        "LLM judge scores require independent validation."
    )

def render_answer_evaluation():
    """Display saved RAG answer-generation evaluation results."""
    import json
    from pathlib import Path

    import pandas as pd
    import streamlit as st

    st.divider()
    st.header("🧠 RAG Answer Quality Evaluation")

    report_path = Path("data/evaluation/answer_results.json")

    if not report_path.exists():
        st.info(
            "No answer evaluation results found. "
            "Run `python evaluate_answers.py` first."
        )
        return

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        results = report["results"]
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Unable to load answer evaluation: {exc}")
        return

    if not results:
        st.warning("The answer evaluation report contains no results.")
        return

    st.caption(
        f"Model: {report.get('model', 'Unknown')} | "
        f"Evaluation cases: {len(results)}"
    )

    answered = [
        r for r in results
        if r.get("expected_behavior") == "answer"
    ]

    abstentions = [
        r for r in results
        if r.get("expected_behavior") == "abstain"
    ]

    valid_citations = sum(
        bool(r.get("citations_valid", False))
        for r in results
    )

    cited_answers = sum(
        bool(r.get("has_citation", False))
        for r in answered
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Evaluation Cases", len(results))

    col2.metric(
        "Citation Reference Validity",
        f"{valid_citations / len(results):.0%}",
    )

    col3.metric(
        "Citation Coverage",
        f"{cited_answers / len(answered):.0%}"
        if answered else "N/A",
    )

    col4.metric(
        "Abstention Cases",
        len(abstentions),
    )

    st.info(
        "Citation validity checks whether referenced citation numbers "
        "exist in the supplied sources. It does not verify factual "
        "support. Abstention correctness and answer relevance require "
        "independent review."
    )

    st.subheader("Question-Level Results")

    rows = []

    for result in results:
        rows.append({
            "Question": result.get("question", ""),
            "Expected Behavior": result.get("expected_behavior", ""),
            "Valid Citations": result.get("citations_valid", False),
            "Has Citation": result.get("has_citation", False),
            "Citations": ", ".join(
                str(c) for c in result.get("citations", [])
            ) or "None",
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.subheader("Generated Answers")

    for index, result in enumerate(results, start=1):
        question = result.get("question", f"Question {index}")

        with st.expander(f"{index}. {question}"):
            st.markdown("**Generated Answer**")
            st.write(result.get("answer", ""))

            st.markdown("**Expected Behavior**")
            st.write(result.get("expected_behavior", "Unknown"))

            st.markdown("**Source References**")
            sources = result.get("sources", [])

            if sources:
                for source in sources:
                    st.write(
                        f"[{source.get('citation', '?')}] "
                        f"{source.get('source', 'Unknown')}"
                    )
            else:
                st.caption("No retrieved sources.")

            st.markdown("**Manual Review**")

            review = result.get("manual_review", {})

            st.write({
                "Answer relevant": review.get("answer_relevant"),
                "Factually grounded": review.get("factually_grounded"),
                "Correct abstention": review.get("correct_abstention"),
            })

    st.caption(
        "Evaluation scope: controlled answer-generation examples. "
        "This report does not establish end-to-end RAG accuracy "
        "or a general hallucination rate."
    )

    render_quality_evaluation()
