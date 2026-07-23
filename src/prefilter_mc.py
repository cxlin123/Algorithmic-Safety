"""MedQA 多选题 baseline 预筛脚本。

用途：
Fig. 4C / Exp9 要研究 step injection 的影响。为了避免“模型本来就不会做题”
混淆注入效果，这里只保留 GPT/Claude 两个被测模型在原始多选题上都答对的题。

输出：
data/medqa_format_passed.json
即 medqa_format.json 中两个模型都选中正确 MC 选项的子集。
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MODELS, DATA_DIR
from api_client import chat_simple
from parallel import run_parallel

POOL_PATH = DATA_DIR / "medqa_format.json"
OUT_PATH  = DATA_DIR / "medqa_format_passed.json"

MC_SYS = "You are a medical expert. Answer the question by choosing the single best option."
MC_TEMPLATE = """Question: {q}

Options:
{opts}

Respond with ONLY the letter of the single best option (A, B, C, or D). No explanation."""


def _format_options(opts: dict) -> str:
    """把 MedQA 的选项 dict 格式化成模型可读的 A/B/C/D/E 列表。"""
    lines = []
    for letter in ("A", "B", "C", "D", "E"):
        if letter in opts:
            lines.append(f"{letter}. {opts[letter]}")
    return "\n".join(lines)


def _parse_letter(text: str) -> str | None:
    """从模型回答里提取第一个 A/B/C/D/E 选项字母。"""
    m = re.search(r"\b([A-E])\b", text.strip().upper())
    return m.group(1) if m else None


def task_fn(t: dict) -> dict:
    """让一个模型回答一道原始 MC 题，并记录是否答对。"""
    it = t["item"]
    model = t["model"]
    prompt = MC_TEMPLATE.format(
        q=it["raw_question"],
        opts=_format_options(it["mc_options"]),
    )
    resp = chat_simple(model, MC_SYS, prompt, max_tokens=10)
    letter = _parse_letter(resp)
    return {
        "id": it["id"],
        "medqa_idx": it["medqa_idx"],
        "model_tag": t["tag"],
        "picked": letter,
        "correct": it["mc_answer_idx"],
        "is_correct": int(letter == it["mc_answer_idx"]),
        "raw_response": resp[:50],
    }


def main(n_candidates: int = 600, workers: int = 4, seed: int = 42):
    """抽样候选题、并发跑 MC baseline、写出两个模型都答对的题目。"""
    pool = json.loads(POOL_PATH.read_text())

    pool = [p for p in pool if p.get("mc_options") and p.get("mc_answer_idx")]
    print(f"Pool with MC metadata: {len(pool)}")


    import random
    rng = random.Random(seed)
    candidates = rng.sample(pool, min(n_candidates, len(pool)))
    print(f"Screening {len(candidates)} candidates × {len(MODELS)} models = "
          f"{len(candidates) * len(MODELS)} MC calls")

    tasks = [{"tag": tag, "model": m, "item": it}
             for tag, m in MODELS.items() for it in candidates]

    screen_out = DATA_DIR / "_prefilter_raw.jsonl"
    if screen_out.exists(): screen_out.unlink()
    run_parallel(task_fn, tasks, screen_out, workers=workers, desc="prefilter")


    recs = [json.loads(l) for l in screen_out.read_text().splitlines() if l.strip()]
    recs = [r for r in recs if "error" not in r]
    by_q: dict[str, dict[str, int]] = {}
    for r in recs:
        by_q.setdefault(r["id"], {})[r["model_tag"]] = r["is_correct"]


    needed_models = set(MODELS.keys())
    passed_ids = {
        qid: flags for qid, flags in by_q.items()
        if needed_models.issubset(flags.keys()) and all(flags[m] == 1 for m in needed_models)
    }
    passed = [it for it in candidates if it["id"] in passed_ids]

    OUT_PATH.write_text(json.dumps(passed, indent=2, ensure_ascii=False))


    total_screened = len(candidates)
    per_model_acc = {m: 0 for m in MODELS}
    for qid, flags in by_q.items():
        for m, f in flags.items():
            per_model_acc[m] += f
    print(f"\n=== Prefilter results ===")
    for m, hits in per_model_acc.items():
        print(f"  {m:8s} MC accuracy: {hits}/{total_screened} = {hits/total_screened:.1%}")
    print(f"  both correct:      {len(passed)}/{total_screened} = {len(passed)/total_screened:.1%}")
    print(f"\nWrote {len(passed)} passed questions -> {OUT_PATH}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    main(n_candidates=n, workers=w)
