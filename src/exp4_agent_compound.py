"""Exp4：五步医疗 agent 的 compound p_safe 实验。

对应论文 Figure 4A。
每道题运行一次五步 ReAct-style agent，每一步都由 Gemini/Qwen judge 面板给出
p_safe，最后把五步 p_safe 相乘得到 compound_psafe。
"""
from __future__ import annotations
from config import MODELS, RESULTS_DIR
from data_loader_medqa_format import load_questions
from agent import run_agent
from parallel import run_parallel


def task_fn(t: dict) -> dict:
    """运行一个模型-题目 agent trace，并保存逐步 judge 结果。"""
    trace = run_agent(t["model"], t["item"]["question"])
    it = t["item"]

    step_records = [
        {
            "name": s["name"],
            "input": s["in"][:2000],
            "output": s["out"][:2000],
            "judge": s["judge"],
            "tool": s.get("tool"),
            "irreversibility": s.get("irreversibility"),
        }
        for s in trace["steps"]
    ]
    return {
        "model_tag": t["tag"], "model": t["model"],
        "id": it["id"],
        "per_step_psafe": trace["per_step_psafe"],
        "compound_psafe": trace["compound_psafe"],
        "step_judges": [s["judge"] for s in trace["steps"]],
        "step_records": step_records,
    }


def main(n: int = 100, workers: int = 16):
    """构造 Exp4 任务并写出 results/exp4_agent_compound.jsonl。"""
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
