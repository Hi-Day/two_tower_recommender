"""
Brute Force Retrieval Index
===========================

Exact nearest-neighbor retrieval.

Reference implementation for all retrieval backends.
"""

from __future__ import annotations

import numpy as np

from .index import BaseIndex
from .search_result import SearchResult


class BruteForceIndex(BaseIndex):
    """
    Exact nearest-neighbor search.

    This implementation compares every query against
    every indexed item.

    Mainly intended for:

    - Evaluation
    - Small datasets
    - Unit testing
    - Baseline retrieval
    """

    def __init__(self):

        self.embeddings: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None

    # -----------------------------------------------------

    @property
    def size(self) -> int:

        if self.item_ids is None:
            return 0

        return len(self.item_ids)

    # -----------------------------------------------------

    @property
    def dimension(self) -> int:

        if self.embeddings is None:
            return 0

        return self.embeddings.shape[1]

    # -----------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        item_ids: np.ndarray,
    ) -> None:

        if len(embeddings) != len(item_ids):
            raise ValueError(
                "embeddings and item_ids must have the same length."
            )

        self.embeddings = embeddings.astype(
            np.float32,
            copy=False,
        )

        self.item_ids = item_ids

    # -----------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        top_k: int = 10,
    ) -> RecommendationResult:

        if self.embeddings is None:
            raise RuntimeError(
                "Index has not been built."
            )

        query = np.asarray(
            query,
            dtype=np.float32,
        )

        query = np.squeeze(query)

        scores = self.embeddings @ query

        top_k = min(
            top_k,
            self.size,
        )

        indices = np.argpartition(
            scores,
            -top_k,
        )[-top_k:]

        indices = indices[
            np.argsort(
                scores[indices]
            )[::-1]
        ]

        return SearchResult(
            indices=indices,
            item_ids=self.item_ids[indices],
            scores=scores[indices],
        )

    # -----------------------------------------------------

    def save(
        self,
        path: str,
    ) -> None:

        if self.embeddings is None:
            raise RuntimeError(
                "Nothing to save."
            )

        np.savez_compressed(

            path,

            embeddings=self.embeddings,

            item_ids=self.item_ids,

        )

    # -----------------------------------------------------

    def load(
        self,
        path: str,
    ) -> None:

        data = np.load(path)

        self.embeddings = data["embeddings"]

        self.item_ids = data["item_ids"]

    # -----------------------------------------------------

    def __len__(self):

        return self.size

    # -----------------------------------------------------

    def __repr__(self):

        return (
            f"BruteForceIndex("
            f"size={self.size}, "
            f"dimension={self.dimension})"
        )