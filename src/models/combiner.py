"""
Feature Combiner Framework
==========================

Feature combiners aggregate multiple feature embeddings into
a single representation before the projection head.

Author
------
Hidayaturrahman

Compatible
----------
TensorFlow 2.19
Keras 3
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

import tensorflow as tf

from src.core.registry import Registry
from src.core.types import CombinedEmbedding

# ==========================================================
# Registry
# ==========================================================

COMBINERS = Registry()

# ==========================================================
# Type Alias
# ==========================================================

EmbeddingDict = Dict[str, tf.Tensor]


# ==========================================================
# Base Combiner
# ==========================================================

class BaseCombiner(
    tf.keras.layers.Layer,
    ABC,
):
    """
    Base class for all feature combiners.

    Parameters
    ----------
    feature_names:
        Ordered list of feature names.

    Example
    -------
    feature_names =

    [
        "movie_id",
        "genre",
        "year",
    ]
    """

    def __init__(
        self,
        feature_names: List[str],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.feature_names = list(feature_names)

    # ------------------------------------------------------

    @property
    def num_features(self) -> int:
        return len(self.feature_names)

    # ------------------------------------------------------

    def validate_inputs(
        self,
        embeddings: EmbeddingDict,
    ) -> None:
        """
        Validate input embeddings.
        """

        if not embeddings:
            raise ValueError(
                "Embedding dictionary is empty."
            )

        missing = [

            feature

            for feature in self.feature_names

            if feature not in embeddings

        ]

        if missing:

            raise KeyError(

                f"Missing embedding(s): {missing}"

            )

    # ------------------------------------------------------

    def ordered_embeddings(
        self,
        embeddings: EmbeddingDict,
    ) -> List[tf.Tensor]:
        """
        Return embeddings following feature_names order.
        """

        self.validate_inputs(embeddings)

        tensors = []

        for feature in self.feature_names:

            tensors.append(
                embeddings[feature]
            )

        return tensors

    # ------------------------------------------------------

    def feature_shapes(
        self,
        embeddings: EmbeddingDict,
    ) -> Dict[str, tf.TensorShape]:
        """
        Return feature shapes.
        Useful for debugging.
        """

        return {

            feature: embeddings[feature].shape

            for feature in self.feature_names

            if feature in embeddings

        }

    # ------------------------------------------------------

    def diagnostics(
        self,
        embeddings: EmbeddingDict,
    ) -> dict:
        """
        Default diagnostics.

        Child classes can override.
        """

        return {

            "strategy": self.__class__.__name__,

            "num_features": self.num_features,

            "feature_shapes":

                self.feature_shapes(
                    embeddings
                ),

        }

    # ------------------------------------------------------

    @abstractmethod
    def call(
        self,
        embeddings: EmbeddingDict,
    ) -> CombinedEmbedding:
        """
        Combine feature embeddings.

        Returns
        -------
        CombinedEmbedding
        """
        ...

    # ------------------------------------------------------

    def get_config(self):

        config = super().get_config()

        config.update(

            {

                "feature_names":

                    self.feature_names,

            }

        )

        return config


# ==========================================================
# Helper
# ==========================================================

def stack_embeddings(
    tensors: List[tf.Tensor],
) -> tf.Tensor:
    """
    Stack embeddings.

    Input

    [
        (B,D),
        (B,D),
        (B,D)
    ]

    Output

    (B,F,D)

    where

    F = number of features
    """

    return tf.stack(
        tensors,
        axis=1,
    )


# ==========================================================
# Concat Combiner
# ==========================================================

@COMBINERS.register("concat")
class ConcatCombiner(
    BaseCombiner,
):
    """
    Concatenate feature embeddings.

    Example

    movie_id -> (B,64)

    genre -> (B,64)

    year -> (B,64)

    Output

    (B,192)
    """

    def call(
        self,
        embeddings: EmbeddingDict,
    ) -> CombinedEmbedding:

        tensors = self.ordered_embeddings(
            embeddings
        )

        if len(tensors) == 1:

            embedding = tensors[0]

        else:

            embedding = tf.concat(

                tensors,

                axis=-1,

            )

        return CombinedEmbedding(

            embedding=embedding,

            diagnostics=

                self.diagnostics(
                    embeddings
                ),

        )

# ==========================================================
# Reduce Combiner
# ==========================================================


class ReduceCombiner(BaseCombiner):
    """
    Base class for reducers operating on stacked embeddings.

    Input:
        (B,F,D)

    Output:
        (B,D)
    """

    @abstractmethod
    def reduce(
        self,
        stacked: tf.Tensor,
    ) -> tf.Tensor:
        """
        Reduce stacked embeddings.

        Parameters
        ----------
        stacked:
            shape = (batch, features, embedding_dim)
        """
        ...

    def call(
        self,
        embeddings: EmbeddingDict,
    ) -> CombinedEmbedding:

        tensors = self.ordered_embeddings(
            embeddings
        )

        if len(tensors) == 1:

            embedding = tensors[0]

        else:

            stacked = stack_embeddings(
                tensors
            )

            embedding = self.reduce(
                stacked
            )

        return CombinedEmbedding(

            embedding=embedding,

            diagnostics=self.diagnostics(
                embeddings
            ),

        )

# ==========================================================
# Mean Combiner
# ==========================================================


@COMBINERS.register("mean")
class MeanCombiner(
    ReduceCombiner,
):
    """
    Mean pooling.
    """

    def reduce(
        self,
        stacked: tf.Tensor,
    ) -> tf.Tensor:

        return tf.reduce_mean(

            stacked,

            axis=1,

        )

# ==========================================================
# Sum Combiner
# ==========================================================


@COMBINERS.register("sum")
class SumCombiner(
    ReduceCombiner,
):
    """
    Sum pooling.
    """

    def reduce(
        self,
        stacked,
    ):

        return tf.reduce_sum(

            stacked,

            axis=1,

        )

# ==========================================================
# Max Combiner
# ==========================================================


@COMBINERS.register("max")
class MaxCombiner(
    ReduceCombiner,
):
    """
    Max pooling.
    """

    def reduce(
        self,
        stacked,
    ):

        return tf.reduce_max(

            stacked,

            axis=1,

        )

# ==========================================================
# Attention Combiner
# ==========================================================


class AttentionCombiner(BaseCombiner):
    """
    Base class for attention-based feature aggregation.
    """

    @abstractmethod
    def compute_attention(
        self,
        stacked: tf.Tensor,
    ) -> tf.Tensor:
        """
        Return attention weights.

        Output shape

        (features,)
        """
        ...

    def call(
        self,
        embeddings: EmbeddingDict,
    ) -> CombinedEmbedding:

        tensors = self.ordered_embeddings(
            embeddings
        )

        if len(tensors) == 1:

            return CombinedEmbedding(

                embedding=tensors[0],

                diagnostics=self.diagnostics(
                    embeddings
                ),

            )

        stacked = stack_embeddings(
            tensors
        )

        weights = self.compute_attention(
            stacked
        )

        embedding = tf.einsum(
            "f,bfd->bd",
            weights,
            stacked,
        )

        diagnostics = self.diagnostics(
            embeddings
        )

        diagnostics["feature_weights"] = {

            feature: float(weight)

            for feature, weight in zip(

                self.feature_names,

                weights.numpy(),

            )

        }

        return CombinedEmbedding(

            embedding=embedding,

            diagnostics=diagnostics,

        )


# ==========================================================
# Weighted Mean Combiner
# ==========================================================


@COMBINERS.register("weighted")
class WeightedMeanCombiner(
    AttentionCombiner,
):
    """
    Learnable weighted feature aggregation.
    """

    def build(
        self,
        input_shape,
    ):

        self.logits = self.add_weight(

            name="feature_logits",

            shape=(self.num_features,),

            initializer="zeros",

            trainable=True,

        )

    def compute_attention(
        self,
        stacked,
    ):

        return tf.nn.softmax(

            self.logits

        )

    @property
    def feature_weights(self):

        weights = tf.nn.softmax(
            self.logits
        ).numpy()

        return {

            feature: float(weight)

            for feature, weight in zip(

                self.feature_names,

                weights,

            )

        }

    def get_config(self):

        config = super().get_config()

        return config


