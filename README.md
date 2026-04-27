# Safety as a System Property

We audit two frontier LLMs (GPT-5 and Claude-Sonnet-4.6) on clinical vignettes derived from MedQA-USMLE and produce the three empirical figures in the paper:

| Paper figure | Principle | What it shows |
|---|---|---|
| Figure 3 | P1 — Cross-class stress testing | A single jailbreak can concurrently degrade hallucination, harmful-content, and age-related prescribing bias metrics |
| Figure 4 | P2 — Irreversibility-aware design | Compound safety leakage in a 5-step ReAct medical agent; C1→C4 attack collapse; step-targeted failure injection |
| Figure 5 | P3 — Calibrated transparency | Exposing the safety rubric raises the judge's score while real harm also rises |

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure API access (OpenAI-compatible endpoint)
export API_BASE="https://api.openai.com/v1"
export API_KEY="sk-..."

# 3. Run all experiments and render figures
bash run_all.sh
```

Edit `src/config.py` to change model identifiers if your endpoint uses different names.

## Source dataset

The evaluation set is derived from [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) (Jin et al., 2020; [arXiv:2009.13081](https://arxiv.org/abs/2009.13081); [original repo](https://github.com/jind11/MedQA)). Questions are expert-curated USMLE-style clinical vignettes, not real hospital records.

We apply a deterministic structural filter (character length, clinical-demographic opener, vignette markers, image independence, identifiable MC tail) and replace each trailing multiple-choice prompt with a uniform open-ended instruction. All preceding clinical content is preserved verbatim. The pipeline is fully reproducible:

```bash
# Reproduce the question pool from the raw HuggingFace download
python src/data_loader_medqa_format.py
```

## Repository layout

```
├── src/
│   ├── config.py                        # API endpoints, model IDs, paths
│   ├── data_loader_medqa_format.py      # Deterministic structural filter + samplers
│   ├── prefilter_mc.py                  # MC-accuracy pre-filter (Fig 4C only)
│   ├── jailbreaks.py                    # Five jailbreak conditions (Baseline/DAN/Role Play/PAIR/Indirect)
│   ├── judge.py                         # LLM-as-judge rubrics
│   ├── agent.py                         # 5-step ReAct medical agent
│   ├── exp1_c1_to_c2.py                 # Fig 3 — hallucination & harmful-content rates
│   ├── exp2_c1_to_c3.py                 # Fig 3 — age-stratified prescribing bias
│   ├── exp4_agent_compound.py           # Fig 4A — cumulative P_safe
│   ├── exp5_agent_jailbreak.py          # Fig 4B — clean vs. attacked
│   ├── exp8_principle3_transparency.py  # Fig 5 — rubric-disclosure paradox
│   ├── exp9_step_injection.py           # Fig 4C — step-targeted failure injection
│   ├── calibrate_payloads.py            # One-time severity calibration for injection payloads
│   ├── fig1_2models.py                  # Renders Figure 3
│   ├── fig2_2models.py                  # Renders Figure 4
│   └── fig3_2models.py                  # Renders Figure 5
├── data/
│   ├── medqa_format.json                # Filtered vignette pool (main)
│   └── medqa_format_passed.json         # MC-correct subset (Fig 4C)
├── results/                             # Per-experiment JSONL outputs
└── figures/                             # Rendered PDF/PNG figures
```

## Running individual experiments

Each experiment script takes `(N, workers)` as positional arguments and streams results to `results/` as JSONL, so runs can be safely interrupted and resumed.

```bash
# Figure 3 — cross-class stress test
python src/exp1_c1_to_c2.py              100 4   # hallucination & harmful rates
python src/exp2_c1_to_c3.py              100 4   # age-stratified bias (per cohort)

# Figure 4 — compound agent risk
python src/exp4_agent_compound.py        100 4   # Panel A: cumulative P_safe
python src/exp5_agent_jailbreak.py       100 4   # Panel B: clean vs. attacked
python src/prefilter_mc.py               600 4   # Panel C prerequisite: MC pre-filter
python src/calibrate_payloads.py                 # Panel C prerequisite: severity check
python src/exp9_step_injection.py         80 4   # Panel C: step-targeted injection

# Figure 5 — transparency paradox
python src/exp8_principle3_transparency.py 100 4
```

### Render figures

```bash
python src/fig1_2models.py   # → figures/fig1_2models.{png,pdf}  (paper Figure 3)
python src/fig2_2models.py   # → figures/fig2_2models.{png,pdf}  (paper Figure 4)
python src/fig3_2models.py   # → figures/fig3_2models.{png,pdf}  (paper Figure 5)
```

## License

Code is released under the MIT License. The MedQA-USMLE source dataset retains its original license (Jin et al., 2020).
