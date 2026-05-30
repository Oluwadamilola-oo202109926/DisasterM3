# DisasterM3 Repository Analysis

## Current Repository Structure

```
DisasterM3/
├── models/                  # Model definitions (VLM runners)
├── pyscripts/
│   └── run_vllm.py          # Main entry-point: loads dataset subset, runs a VLM, scores outputs
├── __init__.py
└── README.md
```

The repo is intentionally minimal. The primary executable is `pyscripts/run_vllm.py`, which accepts two CLI arguments: `--model_id` (a HuggingFace model identifier) and `--subset` (a DisasterM3 task name such as `bearing_body` or `report`). The `models/` directory likely contains thin wrappers that instantiate the VLM backend.

---

## Code Organisation Analysis

### What works well
- The single entry-point design (`run_vllm.py`) makes it easy to reproduce a run with one command.
- Model selection via `--model_id` is already partially config-driven.
- The codebase is small enough to fully understand in an hour.

### What is lacking
- **No dataset abstraction layer.** The data loading logic is embedded directly in the run script (or in dataset-specific helpers not exposed as classes). There is no `BaseDataset` interface.
- **No evaluation abstraction.** Scoring logic is co-located with inference; swapping the evaluation metric requires editing the runner script.
- **No experiment tracking.** Results are printed or written to ad-hoc log files; there is no integration with MLflow, W&B, or any systematic tracker.
- **No configuration file.** Everything is passed as CLI flags. Running a sweep of models × datasets × tasks requires manual shell scripting.
- **No reusable experiment runner.** Each new experiment is a fresh shell command with no record of hyperparameters, dataset splits, or random seeds.

---

## Is the Framework Tied to a Specific Dataset?

**Yes — it is heavily coupled to DisasterM3.**

Evidence:
1. The `--subset` argument directly names DisasterM3 task splits (`bearing_body`, `report`, etc.). A different dataset would require adding new parsing logic inside the run script.
2. There is no `BaseDataset` class. Any new dataset must be hard-coded into the same script rather than plugged in via an interface.
3. The prompt templates and answer-parsing logic are designed around DisasterM3's multiple-choice and open-ended formats, not a generic QA protocol.

**What it would take to support another dataset (e.g., EarthVQA):**
- Add a dataset-specific loader that reads the JSON QA files and image paths in EarthVQA's format.
- Write or reuse a prompt template compatible with EarthVQA's six question types.
- Add an evaluation function that computes accuracy + RMSE as EarthVQA requires (vs. DisasterM3's accuracy + GPT judge score).
- Modify `run_vllm.py` to dispatch to the right loader based on a `--dataset` argument.
- This is feasible but brittle — every new dataset adds more conditionals to the monolithic script.

---

## Proposed Modular Redesign

The redesign extracts three orthogonal concerns into separate, swappable modules:

```
framework/
├── configs/                  # YAML files — no hardcoded values anywhere
├── datasets/
│   ├── base.py               # BaseDataset(ABC): load() → List[Sample]
│   ├── disasterm3.py         # DisasterM3Dataset(BaseDataset)
│   ├── earthvqa.py           # EarthVQADataset(BaseDataset)
│   └── monitrs.py            # MONITRSDataset(BaseDataset)
├── models/
│   ├── base.py               # BaseModelRunner(ABC): run(sample) → str
│   ├── qwen_runner.py        # QwenRunner(BaseModelRunner)
│   └── internvl_runner.py    # InternVLRunner(BaseModelRunner)
├── evaluation/
│   ├── base.py               # BaseEvaluator(ABC): evaluate(predictions, ground_truth) → dict
│   ├── vqa.py                # VQAEvaluator — accuracy + RMSE
│   └── damage_assessment.py  # DamageEvaluator — IoU-based
├── experiments/
│   ├── runner.py             # ExperimentRunner: orchestrates dataset → model → evaluator
│   └── tracker.py            # MLflow / W&B logging wrapper
├── utils/
└── main.py                   # Reads YAML config, instantiates components, runs experiment
```

**Key design principles:**
1. **Dataset / model decoupling**: `BaseDataset.load()` returns a common `Sample` schema (image paths, question, options, answer). Any model runner consumes the same schema regardless of which dataset produced it.
2. **Config-driven execution**: `main.py` never imports a concrete dataset or model class directly — it reads a YAML config and uses a registry (`datasets.get(name)`, `models.get(name)`) to instantiate the right class. Switching datasets means editing one line in the YAML.
3. **Evaluator registry**: Each dataset registers a default evaluator, but evaluators can also be overridden in the config — so the same model can be scored differently across tasks without touching runner code.
4. **Backward compatibility**: `pyscripts/run_vllm.py` is kept intact and continues to work; the new framework is additive.
