"""
Tower Module
============

A Tower represents one side of a Two-Tower recommender.

Pipeline

Input Features
        │
        ▼
FeatureEncoder
        │
        ▼
FeatureCombiner
        │
        ▼
ProjectionHead
        │
        ▼
Embedding

Compatible
----------
TensorFlow 2.19
Keras 3
"""

from __future__ import annotations

from abc import ABC
from typing import Dict

import tensorflow as tf

from src.core.types import (
    CombinedEmbedding,
    TowerOutput,
)

from .encoder import FeatureEncoder
from .combiner import BaseCombiner
from .projection import BaseProjection


FeatureDict = Dict[str, tf.Tensor]


# ==========================================================
# Base Tower
# ==========================================================


class BaseTower(
    tf.keras.Model,
    ABC,
):
    """
    Base implementation of a recommendation tower.

    Parameters
    ----------
    encoder:
        Feature encoder.

    combiner:
        Feature combiner.

    projection:
        Projection head.
    """

    def __init__(
        self,
        encoder: FeatureEncoder,
        combiner: BaseCombiner,
        projection: BaseProjection,
        name: str | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name,
            **kwargs,
        )

        self.encoder = encoder
        self.combiner = combiner
        self.projection = projection

    # --------------------------------------------------

    @property
    def embedding_dim(self):

        return getattr(
            self.projection,
            "output_dim",
            None,
        )

    # --------------------------------------------------

    def encode_features(
        self,
        features: FeatureDict,
        training: bool = False,
    ) -> CombinedEmbedding:
        """
        Encode raw features into a combined embedding.

        Pipeline

        Features
            ↓
        Encoder
            ↓
        Combiner
        """

        feature_embeddings = self.encoder(
            features,
            training=training,
        )

        combined = self.combiner(
            feature_embeddings,
        )

        return combined

    # --------------------------------------------------

    def project(
        self,
        embedding: tf.Tensor,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Apply projection head.
        """

        return self.projection(
            embedding,
            training=training,
        )

        # --------------------------------------------------

    def call(
        self,
        features: FeatureDict,
        training: bool = False,
    ) -> TowerOutput:
        """
        Forward pass.

        Pipeline

        Raw Features
            ↓
        Feature Encoder
            ↓
        Combiner
            ↓
        Projection
            ↓
        Final Embedding
        """

        combined = self.encode_features(
            features,
            training=training,
        )

        projected = self.project(
            combined.embedding,
            training=training,
        )

        return TowerOutput(
            embedding=projected,
            diagnostics=combined.diagnostics,
        )

    # --------------------------------------------------

    def embed(
        self,
        features: FeatureDict,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Convenience function.

        Returns only the embedding tensor.
        """

        output = self(
            features,
            training=training,
        )

        return output.embedding

    # --------------------------------------------------

    def encode(
        self,
        features: FeatureDict,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Alias of embed().
        """

        return self.embed(
            features,
            training=training,
        )

    # --------------------------------------------------

    def get_config(self):

        config = super().get_config()

        config.update(

            {

                "name": self.name,

            }

        )

        return config

    # --------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config,
    ):
        """
        Keras deserialization.

        Components should be injected manually.
        """

        return cls(**config)

# ==========================================================
# User Tower
# ==========================================================


class UserTower(BaseTower):
    """
    User encoder tower.
    """

    pass


# ==========================================================
# Item Tower
# ==========================================================


class ItemTower(BaseTower):
    """
    Item encoder tower.
    """

    pass