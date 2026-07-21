"""
Recommendation Results
======================
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class RecommendationResult:
    """
    High-level recommendation returned by the engine.
    """

    item_ids: np.ndarray

    scores: np.ndarray

    metadata: list[dict[str, Any]] | None = None

    def __len__(self):
        return len(self.item_ids)

    @property
    def top_score(self):
        return float(self.scores[0])