# Reuse Analysis: EarthVQA

## Overview

EarthVQA (Wang et al., AAAI 2024) is a remote sensing VQA dataset and model framework built on top of the LoveDA land-cover dataset. It provides 6,000 high-resolution satellite images paired with 208,593 QA pairs covering six question types: basic judging, relational judging, basic counting, relational counting, object situation analysis, and comprehensive analysis.

The repository at https://github.com/Junjue-Wang/EarthVQA contains:
- `data/` — dataset loader utilities
- `module/` — the SOBA model (segmentation + VQA)
- `configs/` — YAML configuration files for training runs
- `scripts/` — shell scripts for training and inference
- `train_earthvqa.py` / `predict_soba.py` — top-level entry points

---

## Identified Reusable Design Pattern: Configuration-Driven Dataset Loading

### What it is

EarthVQA uses **YAML configuration files** (in `configs/`) to specify dataset parameters — data root paths, split names, batch sizes, and augmentation settings — separately from the model and training code. The Python loaders read these configs at runtime, meaning no source code needs to be modified to point the pipeline at a different data directory or to switch between train and val splits.

Concretely, the loader receives a `cfg` object hydrated from a YAML file and accesses fields like `cfg.DATA.ROOT`, `cfg.DATA.SPLIT`, and `cfg.DATA.QA_FILE`. The rest of the loading logic (reading JSON annotations, resolving image paths, constructing batches) is generic.

### Why this pattern is reusable

This is precisely the design pattern the proposed modular framework needs. The key insight is **separation of what to load from how to load it**:

- The YAML config encodes *what* (which dataset, which split, which subset of tasks).
- The dataset class encodes *how* (JSON parsing, image I/O, schema normalisation).

Switching between DisasterM3 and EarthVQA requires only a config change, not a code change.

### How it fits into the proposed framework

In the proposed architecture (`framework/configs/`), every experiment is described by a YAML file with a top-level `dataset:` block:

```yaml
# configs/earthvqa_val.yaml
dataset:
  name: earthvqa
  root_dir: /data/EarthVQA
  split: val
  question_types: [basic_judging, relational_counting]
  max_samples: 500

model:
  name: qwen_runner
  model_id: Qwen/Qwen2.5-VL-7B-Instruct

evaluator:
  name: vqa
```

`main.py` reads this file, looks up `dataset.name` in a registry, instantiates `EarthVQADataset(**config["dataset"])`, and passes the resulting `List[Sample]` to the model runner — without any dataset-specific code in the runner or evaluator. This is directly adapted from EarthVQA's config-driven loader design.

### Additional reusable element: JSON annotation schema

EarthVQA's QA files (`Train_QA.json`, `Val_QA.json`, `Test_QA.json`) follow a flat list-of-dicts schema where each entry carries an image reference, a question string, and an answer. DisasterM3 follows the same high-level convention. Defining `BaseDataset.load()` to always return `List[Sample]` — a normalised in-memory representation — is inspired by this shared pattern: both datasets ultimately map their on-disk JSON annotations to (image, question, answer) triples, which is exactly what `Sample` captures.

---

## Summary

| Element | Source in EarthVQA | Adaptation in proposed framework |
|---|---|---|
| YAML config for data path | `configs/*.yaml` → `cfg.DATA.*` | `configs/*.yaml` → `dataset.root_dir` etc. |
| Flat JSON annotation schema | `Train_QA.json` list of dicts | `BaseDataset.load()` → `List[Sample]` |
| Split-aware loader | `split` field in config | `BaseDataset.__init__(split=...)` |
| Question-type filtering | Config or script arg | `subset` / `question_types` in YAML |

EarthVQA's config-driven dataset loading is the single most directly transferable pattern because it solves exactly the dataset-coupling problem identified in the DisasterM3 analysis: by making the dataset selection a config value rather than a code path, the framework becomes trivially extensible to any new dataset without touching runner or evaluator logic.
