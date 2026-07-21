"""
Recommendation Engine
=====================

High-level inference engine.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .index import BaseIndex
from .interfaces import UserEncoder
from .result import RecommendationResult


class RecommendationEngine:

    def __init__(
        self,
        encoder: UserEncoder,
        index: BaseIndex,
    ):
        self.encoder = encoder
        self.index = index

    def encode(
        self,
        features,
    ):

        embedding = self.encoder.encode_user(
            features
        )

        if isinstance(
            embedding,
            tf.Tensor,
        ):
            embedding = embedding.numpy()

        embedding = np.asarray(embedding)

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding[0]

        return np.squeeze(embedding)

    def recommend(
        self,
        features,
        top_k=10,
    ):

        query = self.encode(features)

        result = self.index.search(
            query,
            top_k,
        )

        return RecommendationResult(

            item_ids=result.item_ids,

            scores=result.scores,

            metadata=None,

        )