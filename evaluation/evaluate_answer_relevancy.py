import os
import json
import re
import time
from pathlib import Path
from langchain_groq import ChatGroq

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "evaluation_results.json"
OUTPUT_FILE = "deepeval_results_final.json"

JUDGE_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
SLEEP_SECONDS = 1.0


def load_samples(path: str):
    """Load the 20 evaluation samples produced by evaluate_chatbot_with_contexts.py."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Run evaluate_chatbot_with_contexts.py first."
        )

    data = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("results", "questions", "samples", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(
        "evaluation_results.json must contain a list of samples "
        "or a results/questions/samples/data list."
    )


def extract_field(sample, names, default=""):
    for name in names:
        value = sample.get(name)
        if value is not None and value != "":
            return value
    return default


def parse_judge_response(text: str):
    """Parse the plain-text SCORE/REASON response returned by the judge LLM."""
    text = (text or "").strip()

    score_match = re.search(
        r"SCORE\s*:\s*([01](?:\.\d+)?|\.\d+)",
        text,
        flags=re.IGNORECASE,
    )
    reason_match = re.search(
        r"REASON\s*:\s*(.*)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not score_match:
        fallback = re.search(r"\b(0(?:\.\d+)?|1(?:\.0+)?)\b", text)
        if not fallback:
            raise ValueError(f"No score found in judge response: {text}")
        score = float(fallback.group(1))
    else:
        score = float(score_match.group(1))

    score = max(0.0, min(1.0, score))
    reason = reason_match.group(1).strip() if reason_match else text

    return score, reason


def build_answer_relevancy_prompt(question: str, answer: str):
    return f"""
You are evaluating the Answer Relevancy of an AI assistant response.

Evaluate ONLY whether the response directly addresses the user's question.
Do NOT judge factual correctness and do NOT use outside knowledge.

Scoring:
1.0 = directly and fully relevant to the question
0.75 = mostly relevant, with minor irrelevant or incomplete content
0.50 = partially relevant
0.25 = mostly irrelevant
0.0 = does not address the question

A refusal or statement that information is unavailable CAN still be relevant
if it directly responds to what the user asked.

Return EXACTLY two lines in plain text.
Do not return JSON, markdown, code fences, function calls, or tools.

SCORE: <number from 0 to 1>
REASON: <brief explanation>

QUESTION:
{question}

RESPONSE:
{answer}
""".strip()


def category_means(results):
    grouped = {}

    for row in results:
        score = row.get("answer_relevancy")
        if not isinstance(score, (int, float)):
            continue

        category = row.get("category", "")
        grouped.setdefault(category, []).append(float(score))

    return {
        category: sum(scores) / len(scores)
        for category, scores in grouped.items()
        if scores
    }


def main():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY was not found in the environment. "
            "Set it before running the evaluation."
        )

    samples = load_samples(INPUT_FILE)

    judge = ChatGroq(
        model=JUDGE_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    results = []

    print("=" * 72)
    print("ANSWER RELEVANCY EVALUATION")
    print("=" * 72)
    print(f"Samples: {len(samples)}")
    print(f"Judge model: {JUDGE_MODEL}")

    for i, sample in enumerate(samples, start=1):
        question_id = extract_field(sample, ("id", "question_id"), f"Q{i:02d}")
        category = extract_field(sample, ("category",), "")
        question = extract_field(sample, ("question", "input", "query"), "")
        answer = extract_field(sample, ("answer", "actual_output", "response"), "")

        print(f"[{i}/{len(samples)}] {question_id}")

        if not question or not answer:
            results.append({
                "id": question_id,
                "category": category,
                "question": question,
                "answer": answer,
                "answer_relevancy": None,
                "reason": "Missing question or answer",
                "raw_judge_output": "",
            })
            continue

        try:
            prompt = build_answer_relevancy_prompt(question, answer)
            response = judge.invoke(prompt)
            raw_output = response.content if hasattr(response, "content") else str(response)
            score, reason = parse_judge_response(raw_output)

            results.append({
                "id": question_id,
                "category": category,
                "question": question,
                "answer": answer,
                "answer_relevancy": score,
                "reason": reason,
                "raw_judge_output": raw_output,
            })

            print(f"  Answer Relevancy: {score}")

        except Exception as exc:
            results.append({
                "id": question_id,
                "category": category,
                "question": question,
                "answer": answer,
                "answer_relevancy": None,
                "reason": f"ERROR: {exc}",
                "raw_judge_output": "",
            })
            print(f"  ERROR: {exc}")

        time.sleep(SLEEP_SECONDS)

    valid_scores = [
        float(row["answer_relevancy"])
        for row in results
        if isinstance(row.get("answer_relevancy"), (int, float))
    ]

    output = {
        "samples_expected": len(samples),
        "samples_processed": len(results),
        "valid_scores": len(valid_scores),
        "evaluator_llm": JUDGE_MODEL,
        "metric": "Answer Relevancy",
        "method": "plain-text LLM-as-a-judge; no structured output/tool calling",
        "overall_mean": (
            sum(valid_scores) / len(valid_scores)
            if valid_scores else None
        ),
        "category_means": category_means(results),
        "results": results,
    }

    Path(OUTPUT_FILE).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Valid scores: {len(valid_scores)}/{len(samples)}")
    print(f"Overall mean: {output['overall_mean']}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
