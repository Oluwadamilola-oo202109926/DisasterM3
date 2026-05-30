"""
datasets/earthvqa.py

Dataset adapter for EarthVQA:
    Wang et al., "EarthVQA: Towards Queryable Earth via Relational Reasoning-Based
    Remote Sensing Visual Question Answering", AAAI 2024.
    https://arxiv.org/abs/2312.12222
    https://github.com/Junjue-Wang/EarthVQA

Dataset structure on disk (after downloading from HuggingFace):
    <root_dir>/
    ├── Train/
    │   ├── images_png/      # RGB satellite images (*.png)
    │   └── masks_png/       # Semantic segmentation masks (*.png)
    ├── Val/
    │   ├── images_png/
    │   └── masks_png/
    ├── Test/
    │   └── images_png/      # No masks in test split
    ├── Train_QA.json
    ├── Val_QA.json
    └── Test_QA.json

QA JSON schema (each file is a list of dicts):
    [
      {
        "img_id": "3582",
        "image": "Train/images_png/3582.png",   // relative to root_dir
        "question": "Are there any roads near the water?",
        "answer": "Yes",
        "question_type": "relational_judging",   // one of the six types below
        "scene_type": "urban"                    // "urban" | "rural"
      },
      ...
    ]

Note: In some versions of the release the JSON uses slightly different field names
("id" vs "img_id", "type" vs "question_type"). The loader handles both.
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from .base import BaseDataset, Sample


# The six question types defined in the EarthVQA paper
EARTHVQA_QUESTION_TYPES = [
    "basic_judging",           # Simple yes/no about a single object class
    "relational_judging",      # Yes/no requiring spatial / topological reasoning
    "basic_counting",          # Count instances of a single object class
    "relational_counting",     # Count with a spatial condition (e.g. near water)
    "object_situation",        # Qualitative analysis of an object's state
    "comprehensive_analysis",  # Multi-object summarisation (hardest)
]

# Map split names to subdirectory and QA filename
_SPLIT_MAP = {
    "train": ("Train", "Train_QA.json"),
    "val":   ("Val",   "Val_QA.json"),
    "test":  ("Test",  "Test_QA.json"),
}


class EarthVQADataset(BaseDataset):
    """
    Dataset adapter for EarthVQA.

    Args:
        root_dir:        Path to the dataset root (contains Train/, Val/, Test/ and QA JSONs).
        split:           "train" | "val" | "test"
        question_types:  List of question type strings to include (default: all six types).
        scene_type:      "urban" | "rural" | "all" (default "all")
        max_samples:     If set, truncate loaded list (for quick tests).

    Example:
        ds = EarthVQADataset(
            root_dir="/data/EarthVQA",
            split="val",
            question_types=["relational_judging", "relational_counting"],
        )
        samples = ds.load()
    """

    DATASET_NAME = "earthvqa"

    def __init__(
        self,
        root_dir: str,
        split: str = "val",
        question_types: Optional[List[str]] = None,
        scene_type: str = "all",
        max_samples: Optional[int] = None,
    ):
        super().__init__(root_dir=root_dir, split=split)

        if split not in _SPLIT_MAP:
            raise ValueError(
                f"Unknown EarthVQA split '{split}'. Valid options: {list(_SPLIT_MAP.keys())}"
            )

        invalid_types = (
            set(question_types) - set(EARTHVQA_QUESTION_TYPES)
            if question_types else set()
        )
        if invalid_types:
            raise ValueError(
                f"Unknown question type(s): {invalid_types}. "
                f"Valid options: {EARTHVQA_QUESTION_TYPES}"
            )

        self.question_types = question_types  # None means all
        self.scene_type = scene_type.lower()
        self.max_samples = max_samples

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _qa_path(self) -> Path:
        _, qa_filename = _SPLIT_MAP[self.split]
        return Path(self.root_dir) / qa_filename

    def _resolve_image_path(self, relative: str) -> str:
        return str(Path(self.root_dir) / relative)

    def _resolve_mask_path(self, image_relative: str) -> Optional[str]:
        """
        Derive the mask path from the image path.
        EarthVQA stores masks alongside images with the same filename in masks_png/.
        Test split has no masks.
        """
        if self.split == "test":
            return None
        # e.g. "Train/images_png/3582.png" → "Train/masks_png/3582.png"
        mask_rel = image_relative.replace("images_png", "masks_png")
        full = Path(self.root_dir) / mask_rel
        return str(full) if full.exists() else None

    @staticmethod
    def _normalize_record(record: dict) -> dict:
        """Normalise field name variants across EarthVQA release versions."""
        return {
            "img_id":        str(record.get("img_id") or record.get("id", "")),
            "image":         record.get("image") or record.get("img_path", ""),
            "question":      record.get("question", ""),
            "answer":        record.get("answer"),
            "question_type": (
                record.get("question_type")
                or record.get("type")
                or "unknown"
            ),
            "scene_type":    record.get("scene_type") or record.get("scene", "unknown"),
        }

    def _infer_answer_type(self, qt: str) -> str:
        if qt in ("basic_counting", "relational_counting"):
            return "count"
        if qt == "comprehensive_analysis":
            return "open_ended"
        return "multiple_choice"  # judging and situation are effectively binary / short-answer

    def _record_to_sample(self, record: dict) -> Optional[Sample]:
        r = self._normalize_record(record)

        # Filter by question type
        if self.question_types and r["question_type"] not in self.question_types:
            return None

        # Filter by scene type
        if self.scene_type != "all" and r["scene_type"] != self.scene_type:
            return None

        image_abs = self._resolve_image_path(r["image"])
        mask_abs = self._resolve_mask_path(r["image"])

        return Sample(
            sample_id=f"earthvqa_{r['img_id']}_{r['question_type']}",
            dataset_name=self.DATASET_NAME,
            image_paths=[image_abs],          # single image (not bi-temporal)
            question=r["question"],
            options=None,                     # EarthVQA is open-ended / short-answer
            mask_path=mask_abs,
            answer=r["answer"],
            answer_type=self._infer_answer_type(r["question_type"]),
            task=r["question_type"],
            split=self.split,
            metadata={
                "scene_type": r["scene_type"],
                "img_id": r["img_id"],
            },
        )

    # ─── Public API ──────────────────────────────────────────────────────────

    def load(self) -> List[Sample]:
        """
        Load EarthVQA samples from disk.

        Returns:
            List[Sample]: Loaded samples, optionally filtered by question_type and scene_type.
        """
        qa_path = self._qa_path()
        if not qa_path.exists():
            raise FileNotFoundError(
                f"EarthVQA QA file not found: {qa_path}\n"
                f"Download the dataset from "
                f"https://huggingface.co/datasets/Kingdrone-Junjue/EarthVLSet "
                f"and place it at {self.root_dir}"
            )

        with open(qa_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        samples: List[Sample] = []
        for record in records:
            sample = self._record_to_sample(record)
            if sample is not None:
                samples.append(sample)

        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        return samples

    def __repr__(self) -> str:
        return (
            f"EarthVQADataset("
            f"root_dir='{self.root_dir}', "
            f"split='{self.split}', "
            f"question_types={self.question_types!r}, "
            f"scene_type='{self.scene_type}')"
        )
