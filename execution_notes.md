# DisasterM3 Execution Notes

## Environment & Dependencies

The DisasterM3 repository requires the following to run. These were identified from the README and the `pyscripts/run_vllm.py` entry-point:

### Python Environment
- Python >= 3.9 recommended (VLM backends require modern Python)
- CUDA-capable GPU (at minimum 24 GB VRAM for 7B-parameter models; 80 GB+ for 70B models)

### Key Dependencies
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.40.0
pip install accelerate
pip install vllm            # optional, for faster batch inference
pip install Pillow
pip install tqdm
pip install datasets        # HuggingFace datasets
```

### Model Access
Models are pulled from HuggingFace Hub on first run. Examples used in the README:
- `Qwen/Qwen2.5-VL-7B-Instruct` — requires ~16 GB VRAM (fp16)
- `OpenGVLab/InternVL3-78B` — requires multi-GPU (A100/H100 cluster)

### Dataset Access
The DisasterM3 dataset is gated behind a Google Form:
- Benchmark set: https://forms.gle/APQpmyuThh28HsJdA (released 2025-10-17)
- Instruct set: same form (released 2025-10-23)

After approval, data is downloaded and placed in a path configured in the runner.

---

## Execution Steps

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Junjue-Wang/DisasterM3.git
cd DisasterM3
```

### Step 2 — Install Dependencies
```bash
pip install torch transformers accelerate Pillow tqdm datasets
```

### Step 3 — Obtain the Dataset
Fill the Google Form and download the benchmark set. Unzip and place it according to the expected path (typically set via an env variable or config inside `run_vllm.py`).

### Step 4 — Run a Benchmark Subset
```bash
# Qwen2.5-VL on bearing_body subset
python disaster_m3/pyscripts/run_vllm.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --subset bearing_body

# InternVL3 on report subset
python disaster_m3/pyscripts/run_vllm.py \
    --model_id OpenGVLab/InternVL3-78B \
    --subset report
```

---

## Issues Encountered

### Issue 1 — Dataset Gating
The dataset is not publicly downloadable without filling a request form. This makes fully automated reproducibility impossible; a human approval step is required before any code can actually run end-to-end.

**Workaround**: The framework code can be developed and tested with a small synthetic dataset (a handful of image–question–answer tuples in the same JSON schema) while waiting for data access.

### Issue 2 — GPU Memory Requirements
The smallest usable model (Qwen2.5-VL-7B) needs ~16 GB VRAM in fp16. Running on a consumer GPU (e.g., RTX 3090 24 GB) is possible for 7B models but not for 14B+.

**Workaround**: Use `--load-in-4bit` quantisation via `bitsandbytes` to fit larger models:
```bash
pip install bitsandbytes
# Then pass quantization flag when loading model
```

### Issue 3 — README Missing Setup Details
The README jumps directly to the two run commands without specifying:
- Which directory the script should be run from (`disaster_m3/` root or repo root)
- How to point the script to the downloaded dataset path
- Whether a `requirements.txt` or `environment.yml` is provided (it is not)

**Fix applied to README**: Added a "Setup" section before the "Benchmark" section clarifying the working directory and dataset path configuration.

### Issue 4 — Module Import Path
Running `python disaster_m3/pyscripts/run_vllm.py` from the repo root may fail with `ModuleNotFoundError` if the parent directory is not on `PYTHONPATH`.

**Workaround**:
```bash
export PYTHONPATH=$(pwd):$PYTHONPATH
# or
python -m disaster_m3.pyscripts.run_vllm --model_id ... --subset ...
```

---

## Reproducibility Summary

| Aspect | Status |
|---|---|
| Code is public | ✅ |
| Dependencies specifiable | ✅ (manually identified) |
| Dataset freely downloadable | ❌ (form-gated) |
| `requirements.txt` provided | ❌ (missing — added in this PR) |
| GPU requirements documented | ❌ (missing — noted above) |
| Deterministic results | ⚠️ Depends on VLM sampling temperature |

Overall: **partial reproducibility**. The code structure is clean but the dataset access barrier and missing environment specification make zero-friction reproduction impossible without manual steps.
