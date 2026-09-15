import os
import json
from pathlib import Path
import pandas as pd

from langchain_groq import ChatGroq
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper

# IMPORTANT:
# For Ragas 0.4.3 these legacy metric classes still work with evaluate().
# They emit DeprecationWarning, but the warnings do not stop execution.
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

INPUT_FILE = "evaluation_results.json"
OUTPUT_CSV = "ragas_results_llm_only.csv"
OUTPUT_JSON = "ragas_results_llm_only.json"
SUMMARY_JSON = "ragas_summary_llm_only.json"

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def load_rows(path: str):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το {path}. Βάλε το script στον ίδιο φάκελο "
            "με το evaluation_results.json."
        )

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("Το evaluation_results.json δεν περιέχει έγκυρη λίστα.")

    rows = []
    for item in raw:
        qid = item.get("id", "")
        question = (item.get("question") or "").strip()
        answer = (item.get("answer") or "").strip()
        reference = (item.get("reference_answer") or "").strip()
        contexts = item.get("retrieved_contexts") or []

        if not question:
            raise ValueError(f"{qid}: λείπει question.")
        if not answer:
            raise ValueError(f"{qid}: λείπει answer.")
        if not reference:
            raise ValueError(f"{qid}: λείπει reference_answer.")
        if not isinstance(contexts, list) or not contexts:
            raise ValueError(f"{qid}: λείπουν retrieved_contexts.")

        rows.append({
            "id": qid,
            "category": item.get("category", ""),
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": reference,
        })

    return rows


def build_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            'Δεν βρέθηκε GROQ_API_KEY. Στο PowerShell: '
            '$env:GROQ_API_KEY="ΤΟ_KEY_ΣΟΥ"'
        )

    print(f"Evaluator LLM: {GROQ_MODEL}")

    lc_llm = ChatGroq(
        api_key=api_key,
        model=GROQ_MODEL,
        temperature=0,
        max_tokens=1200,
    )
    return LangchainLLMWrapper(lc_llm)


def main():
    print("=" * 72)
    print("RAGAS EVALUATION - LLM ONLY")
    print("=" * 72)

    rows = load_rows(INPUT_FILE)
    print(f"Loaded samples: {len(rows)}")

    dataset = EvaluationDataset.from_list([
        {
            "user_input": r["user_input"],
            "retrieved_contexts": r["retrieved_contexts"],
            "response": r["response"],
            "reference": r["reference"],
        }
        for r in rows
    ])

    evaluator_llm = build_llm()

    metrics = [
        Faithfulness(llm=evaluator_llm),
        LLMContextPrecisionWithReference(llm=evaluator_llm),
        LLMContextRecall(llm=evaluator_llm),
    ]

    print("\nStarting evaluation...\n")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        batch_size=1,
        raise_exceptions=False,
        show_progress=True,
    )

    df = result.to_pandas()
    df.insert(0, "category", [r["category"] for r in rows])
    df.insert(0, "id", [r["id"] for r in rows])

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    Path(OUTPUT_JSON).write_text(
        df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    base_columns = {
        "id", "category", "user_input",
        "retrieved_contexts", "response", "reference"
    }
    metric_columns = [
        c for c in df.columns
        if c not in base_columns and pd.api.types.is_numeric_dtype(df[c])
    ]

    summary = {
        "samples": len(df),
        "evaluator_llm": GROQ_MODEL,
        "metrics": metric_columns,
        "overall_means": {},
        "category_means": {},
    }

    for col in metric_columns:
        valid = df[col].dropna()
        summary["overall_means"][col] = (
            None if valid.empty else float(valid.mean())
        )

    if metric_columns:
        grouped = df.groupby("category")[metric_columns].mean(numeric_only=True)
        for category, row in grouped.iterrows():
            summary["category_means"][str(category)] = {
                col: None if pd.isna(row[col]) else float(row[col])
                for col in metric_columns
            }

    Path(SUMMARY_JSON).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("RAGAS RESULTS")
    print("=" * 72)

    print("\nOverall means:")
    for metric, value in summary["overall_means"].items():
        print(f"  {metric}: {'N/A' if value is None else f'{value:.4f}'}")

    if metric_columns:
        print("\nMeans by category:")
        print(
            df.groupby("category")[metric_columns]
            .mean(numeric_only=True)
            .round(4)
            .to_string()
        )

    print("\nSaved:")
    print(f"  {OUTPUT_CSV}")
    print(f"  {OUTPUT_JSON}")
    print(f"  {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
