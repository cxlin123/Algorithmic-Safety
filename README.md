# Perspective: Safety as a System Property


| Paper figure | Principle | Main analysis |
|---|---|---|
| Figure 3 | P1 — Cross-class stress testing | Jailbreak effects on hallucination, harmful-content, and age-stratified care metrics |
| Figure 4 | P2 — Irreversibility-aware design | Five-step compound safety, clean-versus-attacked agents, and step-targeted failure injection |
| Figure 5 | P3 — Calibrated transparency | Blind, rubric-shown, and adversarial score-gaming conditions |

## Current repository snapshot

The repository currently includes two processed data files and seven experiment result files.

### Data

| File | Records | Purpose |
|---|---:|---|
| `data/medqa_format.json` | 6,293 | Main filtered vignette pool |
| `data/medqa_format_passed.json` | 522 | Questions passing the multi-model MC pre-filter for Figure 4C |

### Results

Existing rendered artifacts are stored under:

- `figure/figures_rebuttal/`: two-model paper figures
- `figure/figures_llama/`: three-model historical figures
- `figure/figures_with_inference/`: figures with confidence intervals and inference annotations

New runs of the standard figure scripts write to `figures/` by default. Override this with `FIGURES_DIR_OVERRIDE`.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m pip install -r requirements.txt
```

There is currently no `run_all.sh`; experiments must be run individually.

## API and model configuration

All evaluated models and judges use an OpenAI-compatible `/chat/completions` endpoint.

```bash
export XTY_API_BASE="https://api.xty.app/v1"
export XTY_API_KEY="..."
```

`API_BASE` and `API_KEY` are accepted as generic alternatives. The `XTY_*` variables take precedence.

Default model identifiers can be overridden:

```bash
export GPT_MODEL="gpt-5-chat"
export CLAUDE_MODEL="claude-sonnet-4-6"
export GEMINI_JUDGE_MODEL="gemini-2.5-flash-nothinking"
export QWEN_JUDGE_MODEL="qwen-max"
```

The default judge mode is `panel`. Gemini and Qwen receive the same rubric:

- Risk indicators such as `harmful` and `contains_hallucination` use conservative OR aggregation.
- Safety indicators require all available judges to pass.
- Continuous scores use the arithmetic mean.

To run a single judge:

```bash
export JUDGE_MODE="single"
export JUDGE_MODEL="gemini-2.5-flash-nothinking"
```

## Source data and preprocessing

The evaluation pool is derived from [MedQA-USMLE](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options) (Jin et al., 2020; [paper](https://arxiv.org/abs/2009.13081); [original repository](https://github.com/jind11/MedQA)).

These are expert-curated examination vignettes, not hospital records. The preprocessing code:

1. filters questions using deterministic structural rules;
2. removes the trailing multiple-choice question;
3. appends one uniform open-ended clinical-assessment instruction; and
4. preserves the preceding vignette text and original MC metadata.

To rebuild the pool, first place the raw MedQA JSONL at `data/phrases_no_exclude_train.jsonl`, then run:

```bash
python src/data_loader_medqa_format.py
```

Sampling uses fixed random seeds for reproducibility. ID/OOD splits use MedQA's existing `meta_info` field.


## Repository layout

```text
├── src/
│   ├── config.py
│   ├── api_client.py
│   ├── data_loader_medqa_format.py
│   ├── prefilter_mc.py
│   ├── jailbreaks.py
│   ├── judge.py
│   ├── agent.py
│   ├── parallel.py
│   ├── exp1_c1_to_c2.py
│   ├── exp2_c1_to_c3.py
│   ├── exp4_agent_compound.py
│   ├── exp5_agent_jailbreak.py
│   ├── exp7_dist_shift.py
│   ├── exp8_principle3_transparency.py
│   ├── exp9_step_injection.py
│   ├── calibrate_payloads.py
│   ├── fig1_2models.py
│   ├── fig2_2models.py
│   ├── fig3_2models.py
│   └── fig*_ci_preview.py
├── data/
├── results/
├── figure/
│   ├── figures_rebuttal/
│   ├── figures_llama/
│   └── figures_with_inference/
└── requirements.txt
```

## Running experiments

Experiment scripts make live model, judge, and in some agent experiments PubMed requests. They may incur substantial API cost.
## Licensing

The MedQA-USMLE source dataset retains its original terms. This repository currently does not include a standalone `LICENSE` file; add the intended code license before redistribution.
