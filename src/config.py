"""Global config: API, models, paths."""
import os
from pathlib import Path

# OpenAI-compatible API endpoint (e.g., a proxy or the official OpenAI base URL).
# Override via environment variables before running any experiment.
API_BASE = os.environ.get("API_BASE", "https://api.openai.com/v1")
API_KEY  = os.environ.get("API_KEY")
if not API_KEY:
    raise RuntimeError(
        "API_KEY is not set. Export it before running any script:\n"
        "    export API_KEY=sk-..."
    )

# Models under test. Tags are short labels used in JSONL output and figure code.
# Replace identifiers with whatever model names your endpoint exposes.
MODELS = {
    "gpt":    "gpt-5-chat",
    "claude": "claude-sonnet-4-6",
}
JUDGE_MODEL = "gpt-5-chat"

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
for d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Experiment defaults
PILOT_N = 50
FULL_N  = 500
AGENT_N = 100
SEED    = 42
