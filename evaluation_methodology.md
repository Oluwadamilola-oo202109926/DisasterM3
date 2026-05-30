# How Vision-Language Models Are Evaluated

## Nature of Evaluation Data

VLMs are evaluated on **multimodal benchmarks** consisting of:

- **Images** (or image pairs for bi-temporal tasks) — natural photos, satellite imagery, medical scans, etc.
- **Language prompts / questions** — open-ended, multiple-choice, or templated instructions
- **Ground-truth answers** — closed-form labels, free-form text, bounding boxes, or segmentation masks

For remote sensing and disaster-focused VLMs specifically (e.g., DisasterM3, EarthVQA), data takes the form of:

- **Bi-temporal image pairs** (pre- and post-disaster satellite imagery, optical or SAR)
- **Multiple-choice questions** (4–5 options) for recognition, counting, and reasoning tasks
- **Open-ended prompts** for report generation and comprehensive analysis
- **Referring expressions** paired with segmentation masks for grounding tasks

Benchmark splits are typically **held-out test sets** whose ground-truth labels are not released publicly; models submit predictions and receive scores from an evaluation server (e.g., EarthVQA uses CodaBench leaderboards).

---

## Evaluation Metrics

### 1. Classification / Multiple-Choice Accuracy
Used for: disaster type recognition, damage level classification, bearing-body recognition, relational reasoning.

$$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total samples}} \times 100\%$$

The model's output (a chosen option letter or a free-form answer matched against choices) is compared to the ground-truth option.

### 2. RMSE — Root Mean Square Error
Used for: counting tasks (number of damaged buildings, road area estimation).

$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(\hat{y}_i - y_i)^2}$$

Lower is better. RMSE penalises large numerical deviations more than MAE, making it useful for counting where off-by-one and off-by-ten errors are qualitatively different.

### 3. GPT-based Open-Ended Scoring
Used for: disaster captions and restoration advice (free-form text).

A judge LLM (GPT-4.1 in DisasterM3) rates the model's text output on a **1–5 scale** across dimensions such as:
- Damage Assessment Precision (DAP)
- Damage Detail Recall (DDR)
- Factual Correctness (FC)
- Recovery Necessity (RN)
- Strategic Completeness (SC)
- Action Priority Precision (APP)

This approach is needed when there is no single correct answer and human-like nuance is required.

### 4. IoU Metrics — Segmentation Quality
Used for: referring segmentation tasks.

- **cIoU** (cumulative IoU): total intersection / total union over all samples; emphasises large objects
- **mIoU** (mean IoU): average per-sample IoU; treats each sample equally

$$\text{mIoU} = \frac{1}{N}\sum_{i=1}^{N} \frac{|P_i \cap G_i|}{|P_i \cup G_i|}$$

### 5. Overall Accuracy (OA) and Overall Rate (OR)
Used by EarthVQA for holistic comparison across all question types combined.

### 6. AVG — Macro-averaged Score
DisasterM3 reports an AVG metric that averages accuracy across all QA subtasks as a single-number leaderboard score.

---

## Evaluation Paradigms

| Paradigm | Description | Example |
|---|---|---|
| **Closed-form / Multiple-choice** | Model picks from given options | Accuracy (%) |
| **Open-ended generation + LLM judge** | Free text scored by a judge model | GPT-4 score 1–5 |
| **Pixel-level grounding** | Model produces a segmentation mask | cIoU / mIoU |
| **Regression** | Model predicts a number | RMSE |

---

## Important Evaluation Considerations

**Prompt sensitivity**: VLMs can give different answers depending on phrasing. Robust evaluation uses multiple prompt variants and reports variance (as done in DisasterM3 Fig. 9).

**Cross-sensor generalization**: Evaluating separately on optical vs. SAR image inputs reveals how well models handle sensor domain shift.

**Cross-disaster generalization**: Per-disaster-type breakdowns expose bias toward visually simpler disasters (e.g., landslides in rural areas score higher than earthquakes in dense urban areas).

**Baseline comparisons**: Sound benchmarking includes random-chance baselines (for multiple-choice, random = 20% for 5 options) and prior SOTA models so that new results can be contextualised.
