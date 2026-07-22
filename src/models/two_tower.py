"""
Two Tower Model
===============

Overall architecture

             User Tower
                  │
                  ▼
          User Embedding
                  │
                  │
                  ▼
           Retrieval Task
                  ▲
                  │
                  │
          Item Embedding
                  ▲
                  │
             Item Tower

TensorFlow 2.19
Keras 3
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import tensorflow as tf

from .tower import (
    UserTower,
    ItemTower,
)

FeatureDict = Dict[str, tf.Tensor]


# ==========================================================
# Output
# ==========================================================

@dataclass(slots=True)
class TwoTowerOutput:
    """
    Output produced by TwoTowerModel.
    """

    user_embedding: tf.Tensor
    item_embedding: tf.Tensor

    user_diagnostics: dict | None = None
    item_diagnostics: dict | None = None


# ==========================================================
# Model
# ==========================================================

class TwoTowerModel(tf.keras.Model):
    """
    Generic Two-Tower encoder.

    This class is responsible only for producing
    user and item embeddings.

    Any retrieval objective should live outside
    this model.
    """

    def __init__(
        self,
        user_tower: UserTower,
        item_tower: ItemTower,
        name: str = "two_tower",
        **kwargs,
    ):
        super().__init__(
            name=name,
            **kwargs,
        )

        self.user_tower = user_tower
        self.item_tower = item_tower

    # --------------------------------------------------

    @property
    def embedding_dim(self):

        return self.user_tower.embedding_dim

    # --------------------------------------------------


    def similarity(
        self,
        user_embedding: tf.Tensor,
        item_embedding: tf.Tensor,
    ) -> tf.Tensor:
        """
        Dot-product similarity.
        """

        return tf.reduce_sum(

            user_embedding * item_embedding,

            axis=-1,

        )

    # --------------------------------------------------

    def score(
        self,
        user_features: FeatureDict,
        item_features: FeatureDict,
    ) -> tf.Tensor:
        """
        Compute similarity score directly.
        """

        user = self.encode_user(
            user_features
        )

        item = self.encode_item(
            item_features
        )

        return self.similarity(
            user,
            item,
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

    def encode_user(
        self,
        features,
        training=False,
    ):

        return self.user_tower.embed(
            features,
            training=training,
        )

    def encode_item(
        self,
        features,
        training=False,
    ):

        return self.item_tower.embed(
            features,
            training=training,
        )
    
    def call(
        self,
        inputs,
        training=False,
    ):

        return TwoTowerOutput(

            user_embedding=self.encode_user(
                inputs["user"],
                training=training,
            ),

            item_embedding=self.encode_item(
                inputs["item"],
                training=training,
            ),

        )