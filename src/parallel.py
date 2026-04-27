"""Parallel runner: bounded thread pool with progress bar and JSONL streaming."""
from __future__ import annotations
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm

DEFAULT_WORKERS = 16


def run_parallel(fn, tasks: list, out_path: Path, workers: int = DEFAULT_WORKERS,
                  desc: str = "") -> None:
    """fn(task) -> dict; results streamed to out_path as JSONL."""
    lock = threading.Lock()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fout = out_path.open("a")  # append mode for resume
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_safe, fn, t): t for t in tasks}
            for fu in tqdm(as_completed(futs), total=len(futs), desc=desc):
                rec = fu.result()
                with lock:
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
    finally:
        fout.close()


def _safe(fn, task):
    try:
        return fn(task)
    except Exception as e:
        return {"error": str(e), "task": str(task)[:200]}
