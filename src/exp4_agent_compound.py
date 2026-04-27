"""Experiment 4 — Agent compound failure (parallel)."""
from __future__ import annotations
from config import MODELS, RESULTS_DIR
from data_loader_medqa_format import load_questions
from agent import run_agent
from parallel import run_parallel


def task_fn(t: dict) -> dict:
    trace = run_agent(t["model"], t["item"]["question"])
    it = t["item"]
    return {
        "model_tag": t["tag"], "model": t["model"],
        "id": it["id"],
        "per_step_psafe": trace["per_step_psafe"],
        "compound_psafe": trace["compound_psafe"],
        "step_judges": [s["judge"] for s in trace["steps"]],
    }


def main(n: int = 100, workers: int = 16):
    items = load_questions(n)
    tasks = [{"tag": tag, "model": m, "item": it}
             for tag, m in MODELS.items() for it in items]
    out = RESULTS_DIR / "exp4_agent_compound.jsonl"
    if out.exists(): out.unlink()
    run_parallel(task_fn, tasks, out, workers=workers, desc="exp4")
    print(f"saved -> {out}")


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    main(n, w)
