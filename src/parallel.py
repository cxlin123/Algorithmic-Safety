"""并发运行器：线程池执行任务，并以 JSONL 流式写出结果。

中文说明：
原始实验脚本只需要最简单的并发写 JSONL。
Gemini/Qwen judge 调用 API 时成本更高，所以这里保留 resume 支持。
"""
from __future__ import annotations
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

DEFAULT_WORKERS = 16


def _stable_key(value) -> str:
    """把 task/record key 转成稳定字符串，便于 resume 去重。"""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _load_done_keys(out_path: Path, key_fn) -> set[str]:
    """读取已有 JSONL，返回已经完成的 task key。"""
    if not out_path.exists() or key_fn is None:
        return set()
    done = set()
    for line in out_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in rec:
            continue
        done.add(_stable_key(key_fn(rec)))
    return done


def run_parallel(fn, tasks: list, out_path: Path, workers: int = DEFAULT_WORKERS,
                  desc: str = "", key_fn=None, resume: bool = False) -> None:
    """并发执行任务，并把每条结果实时追加写入 JSONL。

    fn：单个任务函数，输入 task，输出 dict。
    key_fn：resume 模式下用于判断任务是否已完成的稳定 key。
    resume：为 True 时跳过已有成功行，只跑缺失任务。
    """
    if resume and key_fn is not None:
        done = _load_done_keys(out_path, key_fn)
        tasks = [t for t in tasks if _stable_key(key_fn(t)) not in done]
        print(f"resume: skipped {len(done)} completed rows, remaining {len(tasks)}")

    lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a")
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_safe, fn, t): t for t in tasks}
            for fu in tqdm(as_completed(futs), total=len(futs), desc=desc):
                rec = fu.result()
                with lock:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
    finally:
        fout.close()


def _safe(fn, task):
    """包装单个任务，确保异常也以 JSONL error 行写入。"""
    try:
        return fn(task)
    except Exception as e:
        return {"error": str(e), "task": str(task)[:200]}
