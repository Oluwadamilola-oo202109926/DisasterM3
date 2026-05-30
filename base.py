"""
datasets/base.py

Abstract base class for all disaster-analysis dataset adapters.
Every dataset in this framework must implement `load()` and return a list of Sample dicts.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Sample:
    """
    Universal sample schema shared across all datasets.

    All dataset adapters must produce instances of this class so that
    model runners and evaluators remain dataset-agnostic.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    sample_id: str                          # Unique identifier within the dataset
    dataset_name: str                       # Source dataset (e.g. "disasterm3", "earthvqa")

    # ── Inputs ────────────────────────────────────────────────────────────────
    image_paths: List[str]                  # Ordered list of image file paths.
                                            # For bi-temporal tasks: [pre_disaster, post_disaster]
                                            # For single-image tasks: [image]
    question: str                           # Natural-language question / instruction

    # ── Optional structured inputs ────────────────────────────────────────────
    options: Optional[List[str]] = None     # Answer choices for multiple-choice questions
    mask_path: Optional[str] = None         # Segmentation mask path (EarthVQA, DisasterM3 seg tasks)
    sensor_type: Optional[str] = None       # "optical" | "SAR" (DisasterM3)

    # ── Ground truth ──────────────────────────────────────────────────────────
    answer: Optional[str] = None            # Correct answer (None at inference time)
    answer_type: Optional[str] = None       # "multiple_choice" | "open_ended" | "count" | "segmentation"

    # ── Task metadata ─────────────────────────────────────────────────────────
    task: Optional[str] = None              # Task name within the dataset (e.g. "bearing_body")
    split: Optional[str] = None            # "train" | "val" | "test"

    # ── Extra dataset-specific fields (not used by framework core) ─────────────
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseDataset(ABC):
    """
    Abstract base class that all dataset adapters must inherit from.

    Subclasses implement:
        - __init__: store config (root_dir, split, tasks, etc.)
        - load():   return List[Sample]

    Example usage:
        dataset = DisasterM3Dataset(root_dir="/data/disasterm3", split="test", subset="bearing_body")
        samples = dataset.load()
    """

    def __init__(self, root_dir: str, split: str = "test", **kwargs):
        """
        Args:
            root_dir: Path to the dataset root directory on disk.
            split:    Dataset split to load ("train", "val", "test").
            **kwargs: Additional keyword arguments for subclass configuration.
        """
        self.root_dir = root_dir
        self.split = split

    @abstractmethod
    def load(self) -> List[Sample]:
        """
        Load the dataset and return a list of Sample objects.

        Returns:
            List[Sample]: A list of samples ready for inference.

        Raises:
            FileNotFoundError: If the dataset root or annotation files do not exist.
            ValueError:        If an unsupported split or task is requested.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        """Return the number of samples (loads the dataset if not yet cached)."""
        if not hasattr(self, "_samples"):
            self._samples = self.load()
        return len(self._samples)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root_dir='{self.root_dir}', "
            f"split='{self.split}')"
        )
