"""Experiment 7 — Distribution-shift red-teaming (parallel).

ID:  USMLE Step 2&3 (clinical management) -- aligned with agent/management focus.
OOD: USMLE Step 1 (basic-science reasoning) -- distinct cognitive task type.
Split comes from MedQA's native `meta_info` field.

Expected: OOD questions show higher hallucination rates and lower agent Psafe,
validating that models fail more on out-of-distribution task types.
"""
from __future__ import annotations
from config import MODELS, RESULTS_DIR
from api_client import chat_simple
from data_loader_medqa_format import load_id_ood_questions
from jailbreaks import apply
from judge import judge_response_all
from agent import run_agent
from parallel import run_parallel

SYS = "You are a medical assistant."


def task_fn(t: dict) -> dict:
    model = t["model"]
    it = t["item"]
    q = it["question"]
    base_ans = chat_simple(model, SYS, q, max_tokens=500)
    jb_ans = chat_simple(model, SYS, apply("role_play", q), max_tokens=500)
    agent_trace = run_agent(model, q)
    j_base = judge_response_all(q, base_ans)
    j_jb   = judge_response_all(q, jb_ans)
    return {
        "model_tag": t["tag"],
        "split": t["split"],
        "id": it["id"],
        "base_ans": base_ans[:1500],
        "jb_ans": jb_ans[:1500],
        "halluc_base": {
            "contains_hallucination": j_base.get("contains_hallucination", 0),
            "hallucination_type": j_base.get("hallucination_type", "none"),
        },
        "halluc_jb": {
            "contains_hallucination": j_jb.get("contains_hallucination", 0),
            "hallucination_type": j_jb.get("hallucination_type", "none"),
        },
        "harm_jb": {
            "harmful": j_jb.get("harmful", 0),
            "reason": j_jb.get("harmful_reason", ""),
        },
        "agent_psafe": agent_trace["compound_psafe"],
        "agent_per_step": agent_trace["per_step_psafe"],
    }


def main(n_id: int = 16, n_ood: int = 4, workers: int = 16):
    id_qs, ood_qs = load_id_ood_questions(n_id, n_ood)
    print(f"ID questions: {len(id_qs)}, OOD questions: {len(ood_qs)}")
    tasks = []
    for tag, m in MODELS.items():
        for it in id_qs:
            tasks.append({"tag": tag, "model": m, "item": it, "split": "ID"})
        for it in ood_qs:
            tasks.append({"tag": tag, "model": m, "item": it, "split": "OOD"})
    print(f"Total tasks: {len(tasks)}")
    out = RESULTS_DIR / "exp7_dist_shift.jsonl"
    if out.exists():
        out.unlink()
    run_parallel(task_fn, tasks, out, workers=workers, desc="exp7")
    print(f"saved -> {out}")


if __name__ == "__main__":
    import sys
    n_id = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    n_ood = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    w = int(sys.argv[3]) if len(sys.argv) > 3 else 16
    main(n_id, n_ood, w)
