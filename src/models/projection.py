"""
Projection Head Framework

Projection heads convert combined feature embeddings into
the final embedding space used for retrieval.

Compatible with:
    TensorFlow 2.19
    Keras 3

Author:
    Hidayaturrahman
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf

from src.core.registry import Registry

# ==========================================================
# Registry
# ==========================================================

PROJECTIONS = Registry()


# ==========================================================
# Base Projection
# ==========================================================


class BaseProjection(
    tf.keras.layers.Layer,
    ABC,
):
    """
    Base class of all projection heads.
    """

    def __init__(
        self,
        output_dim: int,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.normalize = normalize

    # ------------------------------------------------------

    def normalize_output(
        self,
        x: tf.Tensor,
    ) -> tf.Tensor:

        if self.normalize:

            x = tf.math.l2_normalize(
                x,
                axis=-1,
            )

        return x

    # ------------------------------------------------------

    @abstractmethod
    def call(
        self,
        inputs,
        training=False,
    ):
        ...

    # ------------------------------------------------------

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "output_dim": self.output_dim,
                "normalize": self.normalize,
            }
        )

        return config


# ==========================================================
# Identity Projection
# ==========================================================


@PROJECTIONS.register("identity")
class IdentityProjection(
    BaseProjection,
):
    """
    No projection.

    Useful for benchmarking.
    """

    def call(
        self,
        inputs,
        training=False,
    ):

        return self.normalize_output(
            inputs
        )


# ==========================================================
# Linear Projection
# ==========================================================


@PROJECTIONS.register("linear")
class LinearProjection(
    BaseProjection,
):
    """
    Single linear layer.
    """

    def __init__(
        self,
        output_dim,
        use_bias=True,
        **kwargs,
    ):
        super().__init__(
            output_dim,
            **kwargs,
        )

        self.linear = tf.keras.layers.Dense(
            output_dim,
            use_bias=use_bias,
        )

    def call(
        self,
        inputs,
        training=False,
    ):

        x = self.linear(inputs)

        return self.normalize_output(
            x
        )


# ==========================================================
# MLP Projection
# ==========================================================


@PROJECTIONS.register("mlp")
class MLPProjection(
    BaseProjection,
):
    """
    Standard projection head.
    """

    def __init__(
        self,
        output_dim,
        hidden_dim=256,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__(
            output_dim,
            **kwargs,
        )

        self.network = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(hidden_dim),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("gelu"),
                tf.keras.layers.Dropout(dropout),
                tf.keras.layers.Dense(output_dim),
            ]
        )

    def call(
        self,
        inputs,
        training=False,
    ):

        x = self.network(
            inputs,
            training=training,
        )

        return self.normalize_output(
            x
        )


# ==========================================================
# Residual Projection
# ==========================================================


@PROJECTIONS.register("residual")
class ResidualMLPProjection(
    BaseProjection,
):
    """
    Residual MLP projection.
    """

    def __init__(
        self,
        output_dim,
        hidden_dim=256,
        dropout=0.2,
        **kwargs,
    ):
        super().__init__(
            output_dim,
            **kwargs,
        )

        self.fc1 = tf.keras.layers.Dense(
            hidden_dim
        )

        self.bn1 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.layers.Activation(
            "gelu"
        )

        self.drop = tf.keras.layers.Dropout(
            dropout
        )

        self.fc2 = tf.keras.layers.Dense(
            output_dim
        )

        self.skip = tf.keras.layers.Dense(
            output_dim
        )

    def call(
        self,
        inputs,
        training=False,
    ):

        shortcut = self.skip(inputs)

        x = self.fc1(inputs)

        x = self.bn1(
            x,
            training=training,
        )

        x = self.act(x)

        x = self.drop(
            x,
            training=training,
        )

        x = self.fc2(x)

        x = x + shortcut

        return self.normalize_output(
            x
        )


# ==========================================================
# Deep Projection
# ==========================================================


@PROJECTIONS.register("deep")
class DeepProjection(
    BaseProjection,
):
    """
    Multi-layer projection.

    Hidden dimensions example:
        [512,256,128]
    """

    def __init__(
        self,
        output_dim,
        hidden_dims=(512, 256),
        dropout=0.2,
        **kwargs,
    ):
        super().__init__(
            output_dim,
            **kwargs,
        )

        layers = []

        for dim in hidden_dims:

            layers.append(
                tf.keras.layers.Dense(dim)
            )

            layers.append(
                tf.keras.layers.BatchNormalization()
            )

            layers.append(
                tf.keras.layers.Activation(
                    "gelu"
                )
            )

            layers.append(
                tf.keras.layers.Dropout(
                    dropout
                )
            )

        layers.append(
            tf.keras.layers.Dense(output_dim)
        )

        self.network = tf.keras.Sequential(
            layers
        )

    def call(
        self,
        inputs,
        training=False,
    ):

        x = self.network(
            inputs,
            training=training,
        )

        return self.normalize_output(
            x
        )