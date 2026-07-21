from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

class Metric(ABC):
    """Abstract base class for evaluation metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        raise NotImplementedError

    def __call__(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        return self.compute(predictions, targets)

    def get_config(self) -> dict:
        """Return metric configuration."""
        return {}

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def __repr__(self) -> str:
        return self.name

class RecallAtK(Metric):

    def __init__(self, k: int):
        self.k = k

    @property
    def name(self) -> str:
        return f"Recall@{self.k}"

    def get_config(self) -> dict:
        return {
            "k": self.k,
        }

    def compute(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> float:

        predictions = predictions[:, : self.k]

        hits = (
            predictions
            == targets.reshape(-1, 1)
        )

        return float(np.mean(np.any(hits, axis=1)))

class PrecisionAtK(Metric):

    def __init__(self, k: int):
        self.k = k

    @property
    def name(self):
        return f"Precision@{self.k}"

    def get_config(self) -> dict:
        return {
            "k": self.k,
        }

    def compute(self, predictions, targets):

        predictions = predictions[:, : self.k]

        hits = (
            predictions
            == targets.reshape(-1, 1)
        )

        return float(
            np.mean(
                np.sum(hits, axis=1) / self.k
            )
        )

class PrecisionAtK(Metric):

    def __init__(self, k: int):
        self.k = k

    @property
    def name(self):
        return f"Precision@{self.k}"

    def get_config(self) -> dict:
        return {
            "k": self.k,
        }

    def compute(self, predictions, targets):

        predictions = predictions[:, : self.k]

        hits = (
            predictions
            == targets.reshape(-1, 1)
        )

        return float(
            np.mean(
                np.sum(hits, axis=1) / self.k
            )
        )

class HitRateAtK(Metric):

    def __init__(self, k: int):
        self.k = k

    @property
    def name(self):
        return f"HitRate@{self.k}"

    def get_config(self) -> dict:
        return {
            "k": self.k,
        }

    def compute(self, predictions, targets):

        predictions = predictions[:, : self.k]

        hits = (
            predictions
            == targets.reshape(-1, 1)
        )

        return float(
            np.mean(
                np.any(hits, axis=1)
            )
        )

class MRR(Metric):

    @property
    def name(self):
        return "MRR"

    def get_config(self) -> dict:
        return {}

    def compute(self, predictions, targets):

        reciprocal_ranks = []

        for pred, target in zip(predictions, targets):

            indices = np.where(pred == target)[0]

            if len(indices) == 0:
                reciprocal_ranks.append(0.0)
            else:
                reciprocal_ranks.append(
                    1.0 / (indices[0] + 1)
                )

        return float(np.mean(reciprocal_ranks))

class NDCGAtK(Metric):

    def __init__(self, k: int):
        self.k = k

    @property
    def name(self):
        return f"NDCG@{self.k}"

    def get_config(self) -> dict:
        return {
            "k": self.k,
        }

    def compute(self, predictions, targets):

        predictions = predictions[:, : self.k]

        scores = []

        for pred, target in zip(predictions, targets):

            idx = np.where(pred == target)[0]

            if len(idx) == 0:
                scores.append(0.0)
            else:
                scores.append(
                    1.0 / np.log2(idx[0] + 2)
                )

        return float(np.mean(scores))

