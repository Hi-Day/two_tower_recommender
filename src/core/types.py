"""
Common dataclasses used across the framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tensorflow as tf


# ==========================================================
# Combiner Output
# ==========================================================

@dataclass(slots=True)
class CombinedEmbedding:
    """
    Output of a feature combiner.

    Parameters
    ----------
    embedding
        Combined feature embedding.

    diagnostics
        Additional information such as
        attention score,
        feature weights,
        gating score,
        etc.
    """

    embedding: tf.Tensor

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ==========================================================
# Tower Output
# ==========================================================

@dataclass(slots=True)
class TowerOutput:
    """
    Output produced by one tower.

    Useful for debugging.
    """

    embedding: tf.Tensor

    diagnostics: dict[str, Any] = field(
        default_factory=dict
    )


# ==========================================================
# Retrieval Result
# ==========================================================

@dataclass(slots=True)
class RetrievalResult:
    """
    Result returned by ANN search.
    """

    ids: list[str]

    scores: tf.Tensor

    metadata: dict[str, Any] = field(
        default_factory=dict
    )