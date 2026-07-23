import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

DEFAULT_WORKERS = 16


def _stable_key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _load_done_keys(out_path, key_fn):
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


def run_parallel(fn, tasks, out_path, workers=DEFAULT_WORKERS,
                  desc="", key_fn=None, resume=False):
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
    try:
        return fn(task)
    except Exception as e:
        return {"error": str(e), "task": str(task)[:200]}
