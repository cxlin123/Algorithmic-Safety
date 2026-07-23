"""全局配置：API、被测模型、Gemini/Qwen judge、路径。

这个目录是基于原始 Algorithmic-Safety-main 拷贝后改造的版本。
主要变化是：原来的单一 GPT-5 judge 被替换为 Gemini/Qwen 双 judge。
"""
import os
from pathlib import Path



API_BASE = os.environ.get("XTY_API_BASE") or os.environ.get(
    "API_BASE",
    "https://api.xty.app/v1",
)
API_KEY = os.environ.get("XTY_API_KEY") or os.environ.get("API_KEY", "")



MODELS = {
    "gpt": os.environ.get("GPT_MODEL", "gpt-5-chat"),
    "claude": os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
}



JUDGE_MODELS = {
    "gemini": os.environ.get("GEMINI_JUDGE_MODEL", "gemini-2.5-flash-nothinking"),
    "qwen": os.environ.get("QWEN_JUDGE_MODEL", "qwen-max"),
}


JUDGE_MODEL = os.environ.get("JUDGE_MODEL", JUDGE_MODELS["gemini"])
JUDGE_MODE = os.environ.get("JUDGE_MODE", "panel").lower()


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def project_path_from_env(env_name: str, default_dir: str) -> Path:
    """读取目录环境变量；相对路径自动放在当前工程目录下。"""
    raw = os.environ.get(env_name)
    if not raw:
        return ROOT / default_dir
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


RESULTS_DIR = project_path_from_env("RESULTS_DIR_OVERRIDE", "results")
FIGURES_DIR = project_path_from_env("FIGURES_DIR_OVERRIDE", "figures")

for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)


PILOT_N = 50
FULL_N = 500
AGENT_N = 100
SEED = 42
