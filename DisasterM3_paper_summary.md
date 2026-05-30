# DisasterM3 Paper Summary

**Full title**: DisasterM3: A Remote Sensing Vision-Language Dataset for Disaster Damage Assessment and Response  
**Authors**: Junjue Wang*, Weihao Xuan*, et al. (University of Tokyo, RIKEN AIP)  
**Venue**: NeurIPS 2025  
**ArXiv**: https://arxiv.org/abs/2505.21089

---

## Problem Statement

Large vision-language models (VLMs) have made strong progress on general image understanding tasks, and the remote sensing field has begun adapting them for satellite imagery analysis. However, disaster scenarios pose unique challenges that existing datasets and models do not address:

1. **Domain gap**: Disasters involve rarely-seen scenes (collapsed buildings, flooded streets, post-explosion craters) that are underrepresented in general VLM training data.
2. **Multi-sensor complexity**: Extreme weather during disasters often makes optical imagery unusable; Synthetic Aperture Radar (SAR) imagery can see through cloud cover but has very different visual characteristics that models are not trained on.
3. **Task diversity**: Real disaster response requires a spectrum of AI capabilities — recognising what type of disaster occurred, counting damaged buildings, localising destroyed infrastructure, reasoning about spatial relationships between affected objects, and generating actionable reports — none of which existing benchmarks cover end-to-end.
4. **No unified benchmark**: Existing disaster datasets (e.g., FloodNet) cover only a single disaster type with simple tasks, making it impossible to evaluate general-purpose disaster AI.

---

## Solution

The authors introduce **DisasterM3**, a curated multimodal dataset with three defining characteristics indicated by the "M3" suffix:

- **Multi-hazard**: 36 historical disaster events across 10 disaster types (earthquake, flood, wildfire, tornado, tsunami, hurricane, volcano, landslide, conflict, explosion) covering all 5 inhabited continents.
- **Multi-sensor**: Each disaster event has paired pre- and post-disaster **optical** images (WorldView series, 0.8 m resolution) and where available paired post-disaster **SAR** images (Capella Space / Umbra), enabling cross-sensor evaluation.
- **Multi-task**: 9 disaster-related tasks derived from real UNOSAT/FEMA assessment workflows, grouped into five capability areas: recognition, counting, localization (referring segmentation), reasoning, and report generation.

The dataset contains **26,988 bi-temporal image pairs** and **123,010 instruction pairs**. It is split into an Instruct set (for fine-tuning) and a Bench set (for evaluation).

---

## Experimentation

### Models benchmarked (14 total)
- **Open-source general VLMs**: LLaVA-1.5-7B, LLaVA-OV-7B, Kimi-VL-A3B (Instruct and Think variants), InternVL3 (8B / 14B / 78B), Qwen2.5-VL (3B / 7B / 32B / 72B)
- **Commercial models**: GPT-4o, GPT-4.1
- **Remote sensing VLMs**: GeoChat-7B, TeoChat-7B, EarthDial-4B
- **Referring segmentation models**: LISA-7B, PSALM-1.3B, HyperSeg-3B, GeoPixel-7B

### Tasks evaluated
Six QA tasks scored by accuracy (%), two open-ended tasks (disaster caption and restoration advice) scored by a GPT-4.1 judge on a 1–5 scale, and two referring segmentation tasks scored by mIoU / cIoU — across both optical-optical and optical-SAR image settings.

### Key findings
1. Even the best zero-shot models (GPT-4.1, Qwen2.5-VL-72B) score around 40–42% average accuracy on QA tasks — just slightly above chance for some subtasks — confirming the benchmark is genuinely challenging.
2. Remote sensing VLMs (GeoChat, TeoChat, EarthDial) do not outperform general VLMs, showing that existing RS fine-tuning does not transfer to disaster scenarios.
3. Fine-tuning Qwen2.5-VL-7B and InternVL3-8B on the DisasterM3 Instruct set yields consistent improvements (up to +10.4% average accuracy, up to +40.8% referring segmentation mIoU), demonstrating the value of disaster-specific training data.
4. SAR-setting performance is dramatically worse than optical-setting performance across all models, highlighting an open research problem in cross-sensor VLM alignment.
