# LLM Fine-Tuning for UML Antipattern Detection

> Fine-tuning a compact language model to detect structural antipatterns in UML use case diagrams — from synthetic data generation to evaluation.

![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Model](https://img.shields.io/badge/Model-Qwen2.5--Coder--3B-orange)
![Accuracy](https://img.shields.io/badge/Detection%20Accuracy-91.0%25-brightgreen)

---

## Overview

This repository contains the full pipeline for the paper **"Fine-Tuning an LLM for UML Use Case Antipattern Detection"**. It addresses a gap in automated software design quality tooling: detecting structural antipatterns in UML use case models.

The pipeline:

1. **Synthesizes** a labeled dataset of 194 PlantUML diagram pairs (antipattern + refactored counterpart) using Claude Opus across a diverse set of application domains
2. **Fine-tunes** Qwen2.5-Coder-3B-Instruct on this dataset using Low-Rank Adaptation (LoRA, r=8, α=16, 0.48% of parameters)
3. **Evaluates** the fine-tuned model on a held-out, domain-stratified test set

The target antipattern is **Functional Decomposition via the `<<include>>` relationship** — a common misuse of the UML include relationship in which internal implementation steps are modeled as separate use cases rather than as logic internal to a single observable service.

### Results

| Metric | Value |
|---|---|
| Detection accuracy (diagram-level) | **91.0%** |
| Instance-level recall | **0.85** |
| Instance-level precision | **0.79** |
| Instance-level F₁ | **0.82** |
| Training samples | 388 (194 domains × 2) |
| Model parameters | 3B (0.48% fine-tuned) |

---

## Repository structure

```
├── config/
│   ├── antipatterns.yaml       # Antipattern definitions and refactoring strategies
│   └── domains.yaml            # Pool of 194 application domains
│
├── scripts/
│   ├── samples_generate.py     # Generate labeled diagram pairs via Claude API
│   ├── samples_review.py       # GUI: review and approve generated pairs
│   ├── samples_audit.py        # Verify training data integrity
│   └── results_review.py       # GUI: review model evaluation results
│
├── notebooks/
│   └── finetune_eval.ipynb     # Fine-tune and evaluate the model (Unsloth + HuggingFace)
│
└── output/
    └── run_20260325_012118/    # Included dataset — 194 domains, 388 samples
        ├── samples.jsonl       # Training data in chat format (ready for fine-tuning)
        └── domains/            # Per-domain PlantUML source, rendered PNGs, and metadata
```

---

## Included dataset

`output/run_20260325_012118/` contains the complete dataset used in the paper, ready for use without any additional API calls:

- **194 application domains**, each contributing one antipattern sample and one refactored sample
- **`samples.jsonl`** — 388 training samples in chat format (system / user / assistant messages), ready to pass directly to a fine-tuning framework
- **`domains/NNN/`** — per-domain PlantUML source (`.puml`), rendered diagram images (`.png`), and training sample YAML with metadata

To fine-tune on the included dataset, open `notebooks/finetune_eval.ipynb` and point it at `output/run_20260325_012118/samples.jsonl`.

---

## Requirements

- Python 3.12+
- Java (for PlantUML diagram rendering)
- `plantuml.jar` — bundled with the [PlantUML VSCode extension](https://marketplace.visualstudio.com/items?itemName=jebbs.plantuml) or downloadable from [plantuml.com](https://plantuml.com/download)
- An [Anthropic API key](https://console.anthropic.com/) — only required to generate new data with `samples_generate.py`

```bash
uv sync
# or: pip install anthropic pyyaml pillow
```

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY in .env
```

---

## Scripts

### `samples_generate.py` — Synthesize diagram pairs

Calls the Claude API to generate labeled PlantUML use case diagram pairs, renders them to PNG via PlantUML, and writes training samples in JSONL and YAML formats.

```bash
python scripts/samples_generate.py \
  --config config/antipatterns.yaml \
  --domains-config config/domains.yaml \
  --plantuml-jar /path/to/plantuml.jar \
  --output-dir output \
  --num-prompts 194 \
  --sizes small medium \
  --size-weights 0.4 0.6 \
  --task-mode detect \
  --rate-limit 2.5
```

Supports resuming interrupted runs (`--resume`) and regenerating flagged pairs (`--reprocess-flagged`) based on reviewer notes.

**Diagram size tiers:**

| Size | Constructs | Max antipattern instance groups |
|---|---|---|
| Small | 9 – 13 | 1 |
| Medium | 24 – 30 | 3 |
| Large | 31 – 54 | 5 |

---

### `samples_review.py` — Review generated pairs (GUI)

A desktop Tkinter GUI for reviewing generated diagram pairs before training.

```bash
python scripts/samples_review.py output/run_YYYYMMDD_HHMMSS
```

- Side-by-side antipattern / refactored diagram panels with zoom controls
- Mark each pair **Approved**, **Needs Rework**, or **Clear**
- Reviewer notes are passed back to Claude when regenerating flagged pairs
- Progress summary and jump-to-prompt navigation

---

### `samples_audit.py` — Verify training data integrity

Validates all samples in `samples.jsonl` against their source files and checks structural correctness of the antipattern and refactored diagrams.

```bash
python scripts/samples_audit.py output/run_YYYYMMDD_HHMMSS
```

Checks include message content integrity, instance count consistency, element alias resolution, include relationship validity, and refactoring scope (the refactored model must equal the antipattern model with antipattern instances removed and nothing else changed).

---

### `results_review.py` — Review evaluation results (GUI)

A desktop Tkinter GUI for reviewing model predictions after evaluation.

```bash
python scripts/results_review.py output/run_YYYYMMDD_HHMMSS/finetune_eval_<ts>.jsonl
```

Shows each test sample's diagram alongside expected vs. predicted JSON with diff highlighting. Supports per-instance TP/TN/FP/FN classification with manual override, and writes verified metrics back to the review file.

---

## Workflow

```
1. samples_generate.py                               →  synthesize diagram pairs
2. samples_review.py                                 →  review; flag needs-rework pairs
3. samples_generate.py --resume --reprocess-flagged  →  regenerate flagged pairs
   (repeat 2–3 until satisfied)
4. samples_audit.py                                  →  verify data integrity
5. notebooks/finetune_eval.ipynb                     →  fine-tune and evaluate
6. results_review.py                                 →  review per-sample predictions
```

---

## Extending to other antipatterns

The pipeline is designed to generalize. To target a different antipattern:

1. Add an entry to `config/antipatterns.yaml` with a `code`, `name`, `description`, and one or more `refactorings`
2. Run `samples_generate.py` with the updated config
3. Fine-tune as usual with `finetune_eval.ipynb`

---

## Citation

```bibtex
@inproceedings{khan2026llmantipatternsdetection,
  author    = {Yasser Khan and Mohamed El-Attar},
  title     = {Fine-Tuning an {LLM} for {UML} Use Case Antipattern Detection},
  year      = {2026}
}
```

---

## License

MIT
