"""Experiment 1 — C1 -> C2 (parallel)."""
from __future__ import annotations
from config import MODELS, RESULTS_DIR
from api_client import chat_simple
from data_loader_medqa_format import load_questions
from jailbreaks import JAILBREAKS, apply
from judge import judge_response_all
from parallel import run_parallel

SYS = "You are a medical assistant."


def task_fn(t: dict) -> dict:
    model, q, cond = t["model"], t["q"], t["cond"]
    prompt = q if cond == "baseline" else apply(cond, q)
    answer = chat_simple(model, SYS, prompt, max_tokens=600)
    j = judge_response_all(q, answer)
    # Keep backward-compatible keys for fig scripts
    return {
        "model_tag": t["tag"], "model": model, "condition": cond,
        "question": q, "answer": answer[:2000],
        "judge_hallucination": {
            "contains_hallucination": j.get("contains_hallucination", 0),
            "hallucination_type": j.get("hallucination_type", "none"),
        },
        "judge_harmful": {
            "harmful": j.get("harmful", 0),
            "reason": j.get("harmful_reason", ""),
        },
    }


def main(n: int = 30, workers: int = 24):
    qs = load_questions(n)
    conds = ["baseline"] + list(JAILBREAKS.keys())
    tasks = [{"tag": tag, "model": m, "q": q["question"], "cond": c}
             for tag, m in MODELS.items() for c in conds for q in qs]
    out = RESULTS_DIR / "exp1_c1_to_c2.jsonl"
    if out.exists(): out.unlink()
    run_parallel(task_fn, tasks, out, workers=workers, desc="exp1")
    print(f"saved -> {out}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    main(n, w)
