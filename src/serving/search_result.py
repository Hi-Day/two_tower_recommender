"""
Search Results
==============
"""

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SearchResult:
    """
    Raw retrieval results returned by an index.
    """

    indices: np.ndarray

    item_ids: np.ndarray

    scores: np.ndarray

    def __len__(self) -> int:
        return len(self.item_ids)

    @property
    def top_score(self) -> float:
        return float(self.scores[0])