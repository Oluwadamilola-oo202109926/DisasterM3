"""
datasets/disasterm3.py

Dataset adapter for DisasterM3:
    Wang et al., "DisasterM3: A Remote Sensing Vision-Language Dataset
    for Disaster Damage Assessment and Response", NeurIPS 2025.
    https://arxiv.org/abs/2505.21089
    https://github.com/Junjue-Wang/DisasterM3

Dataset structure on disk (after downloading via the official form):
    <root_dir>/
    ├── Bench/
    │   ├── optical/
    │   │   ├── pre/         # pre-disaster optical images  (*.png or *.jpg)
    │   │   └── post/        # post-disaster optical images
    │   ├── sar/
    │   │   ├── pre/
    │   │   └── post/
    │   └── annotations/
    │       ├── bearing_body.json
    │       ├── disaster_type.json
    │       ├── scene_recognition.json
    │       ├── building_counting.json
    │       ├── road_estimation.json
    │       ├── relational_reasoning.json
    │       ├── report.json
    │       └── ...
    └── Instruct/
        └── ...

Each annotation JSON has this schema (inferred from the paper / README):
    [
      {
        "id": "unique_sample_id",
        "pre_image": "relative/path/to/pre.png",
        "post_image": "relative/path/to/post.png",
        "sensor": "optical" | "SAR",
        "question": "...",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],   // null for open-ended tasks
        "answer": "A",
        "task": "bearing_body"
      },
      ...
    ]
"""

import json
import os
from pathlib import Path
from typing import List, Optional

from .base import BaseDataset, Sample


# All task names present in the DisasterM3 benchmark set
DISASTERM3_TASKS = [
    "bearing_body",           # Disaster bearing-body recognition (BBR)
    "disaster_type",          # Disaster type recognition (DTR)
    "scene_recognition",      # Disaster scene recognition (DSR)
    "building_counting",      # Damaged building counting (DBC)
    "road_estimation",        # Damaged road area estimation (DRE)
    "relational_reasoning",   # Object relational reasoning (ORR)
    "report",                 # Comprehensive disaster report generation
]


class DisasterM3Dataset(BaseDataset):
    """
    Dataset adapter for the DisasterM3 benchmark.

    Args:
        root_dir: Path to the dataset root (contains Bench/ and Instruct/ subdirs).
        split:    "bench" (default) | "instruct"
        subset:   Task name to load (see DISASTERM3_TASKS), or None to load all tasks.
        sensor:   "optical" | "sar" | "all" (default "optical")
        max_samples: If set, truncate the loaded list (useful for quick testing).

    Example:
        ds = DisasterM3Dataset(
            root_dir="/data/disasterm3",
            split="bench",
            subset="bearing_body",
            sensor="optical",
        )
        samples = ds.load()
    """

    DATASET_NAME = "disasterm3"

    def __init__(
        self,
        root_dir: str,
        split: str = "bench",
        subset: Optional[str] = None,
        sensor: str = "optical",
        max_samples: Optional[int] = None,
    ):
        super().__init__(root_dir=root_dir, split=split)
        if subset is not None and subset not in DISASTERM3_TASKS:
            raise ValueError(
                f"Unknown DisasterM3 subset '{subset}'. "
                f"Valid options: {DISASTERM3_TASKS}"
            )
        self.subset = subset
        self.sensor = sensor.lower()
        self.max_samples = max_samples

        # Map split name to directory
        self._split_dir_map = {
            "bench": "Bench",
            "instruct": "Instruct",
            "test": "Bench",   # alias
            "train": "Instruct",
        }

    # ─── Internal helpers ────────────────────────────────────────────────────

    def _annotation_dir(self) -> Path:
        split_dir = self._split_dir_map.get(self.split, "Bench")
        return Path(self.root_dir) / split_dir / "annotations"

    def _image_root(self) -> Path:
        split_dir = self._split_dir_map.get(self.split, "Bench")
        return Path(self.root_dir) / split_dir

    def _task_names_to_load(self) -> List[str]:
        if self.subset is not None:
            return [self.subset]
        return DISASTERM3_TASKS

    def _load_annotation_file(self, task: str) -> List[dict]:
        ann_path = self._annotation_dir() / f"{task}.json"
        if not ann_path.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {ann_path}\n"
                f"Make sure you have downloaded the DisasterM3 dataset from "
                f"https://forms.gle/APQpmyuThh28HsJdA and placed it at {self.root_dir}"
            )
        with open(ann_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _resolve_image_path(self, relative_path: str) -> str:
        """Resolve a relative image path from the annotation to an absolute path."""
        return str(self._image_root() / relative_path)

    def _annotation_to_sample(self, record: dict, task: str) -> Sample:
        """Convert a single JSON annotation record to a universal Sample."""
        pre_img = self._resolve_image_path(record["pre_image"])
        post_img = self._resolve_image_path(record["post_image"])

        # Filter by sensor type if requested
        record_sensor = record.get("sensor", "optical").lower()
        if self.sensor != "all" and record_sensor != self.sensor:
            return None  # caller will skip None

        # Determine answer type
        options = record.get("options")
        if task == "report":
            answer_type = "open_ended"
        elif options:
            answer_type = "multiple_choice"
        elif task in ("building_counting", "road_estimation"):
            answer_type = "count"
        else:
            answer_type = "open_ended"

        return Sample(
            sample_id=record.get("id", f"{task}_{record.get('idx', '')}"),
            dataset_name=self.DATASET_NAME,
            image_paths=[pre_img, post_img],
            question=record["question"],
            options=options,
            sensor_type=record_sensor,
            answer=record.get("answer"),
            answer_type=answer_type,
            task=task,
            split=self.split,
            metadata={
                "disaster_type": record.get("disaster_type"),
                "region": record.get("region"),
            },
        )

    # ─── Public API ──────────────────────────────────────────────────────────

    def load(self) -> List[Sample]:
        """
        Load DisasterM3 samples from disk.

        Returns:
            List[Sample]: Loaded samples, filtered by subset and sensor.
        """
        samples: List[Sample] = []

        for task in self._task_names_to_load():
            records = self._load_annotation_file(task)
            for record in records:
                sample = self._annotation_to_sample(record, task)
                if sample is not None:
                    samples.append(sample)

        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        return samples

    def __repr__(self) -> str:
        return (
            f"DisasterM3Dataset("
            f"root_dir='{self.root_dir}', "
            f"split='{self.split}', "
            f"subset={self.subset!r}, "
            f"sensor='{self.sensor}')"
        )
