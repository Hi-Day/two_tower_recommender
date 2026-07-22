"""
Feature Encoder

This module contains reusable encoders for recommendation features.

Author : Hidayaturrahman
"""

from __future__ import annotations

from typing import Dict

import tensorflow as tf


class CategoricalEncoder(tf.keras.layers.Layer):
    """
    Encode one categorical feature using
    StringLookup + Embedding.
    """

    def __init__(
        self,
        vocabulary: list[str],
        embedding_dim: int,
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.lookup = tf.keras.layers.StringLookup(
            vocabulary=vocabulary,
            mask_token=None,
            oov_token="[UNK]",
        )

        self.embedding = tf.keras.layers.Embedding(
            input_dim=len(vocabulary) + 2,
            output_dim=embedding_dim,
            embeddings_initializer="he_uniform",
        )

    def call(self, inputs):

        x = self.lookup(inputs)

        x = self.embedding(x)

        return x


# ----------------------------------------------------------


class NumericEncoder(tf.keras.layers.Layer):
    """
    Encode numeric features.

    Useful for:
        age
        price
        timestamp
        popularity
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 64,
        name: str | None = None,
    ):
        super().__init__(name=name)

        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    hidden_dim,
                    activation="relu",
                ),
                tf.keras.layers.Dense(
                    embedding_dim,
                ),
            ]
        )

    def call(self, inputs):

        x = tf.cast(inputs, tf.float32)

        if len(x.shape) == 1:
            x = tf.expand_dims(x, -1)

        return self.network(x)


# ----------------------------------------------------------


class FeatureEncoder(tf.keras.layers.Layer):
    """
    Generic feature encoder.

    Example
    -------
    vocabularies = {
        "movie_id": [...],
        "genre": [...],
    }

    encoder = FeatureEncoder(
        vocabularies,
        embedding_dim=64
    )

    outputs = encoder(features)
    """

    def __init__(
        self,
        vocabularies: Dict[str, list[str]],
        embedding_dim: int,
        numeric_features: list[str] | None = None,
        name: str = "feature_encoder",
    ):
        super().__init__(name=name)

        if numeric_features is None:
            numeric_features = []

        self.numeric_features = set(numeric_features)

        self.encoders = {}

        #
        # categorical encoders
        #

        for feature, vocab in vocabularies.items():

            self.encoders[feature] = CategoricalEncoder(
                vocabulary=vocab,
                embedding_dim=embedding_dim,
                name=f"{feature}_encoder",
            )

        #
        # numeric encoders
        #

        for feature in numeric_features:

            self.encoders[feature] = NumericEncoder(
                embedding_dim=embedding_dim,
                name=f"{feature}_encoder",
            )

    # ------------------------------------------------------

    def call(
        self,
        features: Dict[str, tf.Tensor],
    ) -> Dict[str, tf.Tensor]:

        outputs = {}

        for feature_name, encoder in self.encoders.items():

            if feature_name not in features:
                continue

            outputs[feature_name] = encoder(
                features[feature_name]
            )

        return outputs

    # ------------------------------------------------------

    @property
    def feature_names(self):

        return list(self.encoders.keys())