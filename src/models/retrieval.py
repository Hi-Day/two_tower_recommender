"""
Retrieval Module
================

Similarity functions for Two-Tower Retrieval.

TensorFlow 2.19
Keras 3
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import tensorflow as tf

from src.core.registry import Registry

# ==========================================================
# Registry
# ==========================================================

SIMILARITIES = Registry()

# ==========================================================
# Base Similarity
# ==========================================================


class BaseSimilarity(
    tf.keras.layers.Layer,
    ABC,
):
    """
    Base similarity function.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.temperature = temperature

    # ------------------------------------------------------

    def scale(
        self,
        score: tf.Tensor,
    ) -> tf.Tensor:
        """
        Apply temperature scaling.
        """

        if self.temperature == 1.0:
            return score

        return score / self.temperature

    # ------------------------------------------------------

    @abstractmethod
    def compute(
        self,
        user_embedding: tf.Tensor,
        item_embedding: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute similarity.
        """
        ...

    @abstractmethod
    def pairwise_compute(
        self,
        user_embedding,
        item_embedding,
    ):
        ...
    # ------------------------------------------------------

    def call(
        self,
        user_embedding: tf.Tensor,
        item_embedding: tf.Tensor,
    ) -> tf.Tensor:

        score = self.compute(
            user_embedding,
            item_embedding,
        )

        return self.scale(score)

    # ------------------------------------------------------

    def pairwise(
        self,
        user_embedding: tf.Tensor,
        item_embedding: tf.Tensor,
    ) -> tf.Tensor:
        """
        Compute similarity matrix.

        user  : (B,D)

        item  : (B,D)

        output

        (B,B)
        """

        score = self.pairwise_compute(
            user_embedding,
            item_embedding,
        )

        return self.scale(score)

    # ------------------------------------------------------

    def get_config(self):

        config = super().get_config()

        config.update(
            {
                "temperature": self.temperature,
            }
        )

        return config


# ==========================================================
# Dot Product
# ==========================================================


@SIMILARITIES.register("dot")
class DotProductSimilarity(
    BaseSimilarity,
):
    """
    Dot product similarity.
    """

    def compute(
        self,
        user_embedding,
        item_embedding,
    ):

        return tf.reduce_sum(

            user_embedding * item_embedding,

            axis=-1,

        )

    
    def pairwise_compute(
        self,
        user_embedding,
        item_embedding,
    ):

        return tf.matmul(
            user_embedding,
            item_embedding,
            transpose_b=True,
        )

# ==========================================================
# Cosine Similarity
# ==========================================================


@SIMILARITIES.register("cosine")
class CosineSimilarity(
    BaseSimilarity,
):
    """
    Cosine similarity.
    """

    def compute(
        self,
        user_embedding,
        item_embedding,
    ):

        user_embedding = tf.math.l2_normalize(
            user_embedding,
            axis=-1,
        )

        item_embedding = tf.math.l2_normalize(
            item_embedding,
            axis=-1,
        )

        return tf.reduce_sum(

            user_embedding * item_embedding,

            axis=-1,

        )

    def pairwise_compute(
        self,
        user_embedding,
        item_embedding,
    ):

        user_embedding = tf.math.l2_normalize(
            user_embedding,
            axis=-1,
        )

        item_embedding = tf.math.l2_normalize(
            item_embedding,
            axis=-1,
        )

        return tf.matmul(
            user_embedding,
            item_embedding,
            transpose_b=True,
        )

# ==========================================================
# Retrieval Task
# ==========================================================


class RetrievalTask (
    tf.keras.layers.Layer,
):
    """
    Generic retrieval objective.

    Uses in-batch negatives.

    Positive pair

        user_i <-> item_i

    Negative pair

        user_i <-> item_j
    """

    def __init__(
        self,
        similarity: BaseSimilarity,
        from_logits: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.similarity = similarity

        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=from_logits,
        )

    # ------------------------------------------------------

    def similarity_matrix(
        self,
        user_embedding: tf.Tensor,
        item_embedding: tf.Tensor,
    ) -> tf.Tensor:
        """
        Build similarity matrix.

        Shape

            (B,B)
        """

        return self.similarity.pairwise(
            user_embedding,
            item_embedding,
        )

    # ------------------------------------------------------

    def labels(
        self,
        batch_size: tf.Tensor,
    ) -> tf.Tensor:
        """
        Positive labels.

        user_i -> item_i
        """

        return tf.range(
            batch_size,
            dtype=tf.int32,
        )

    # ------------------------------------------------------

    def compute_loss(
        self,
        similarity_matrix: tf.Tensor,
    ) -> tf.Tensor:
        """
        Cross entropy retrieval loss.
        """

        batch_size = tf.shape(
            similarity_matrix
        )[0]

        labels = self.labels(
            batch_size
        )

        return self.loss_fn(
            labels,
            similarity_matrix,
        )

    # ------------------------------------------------------

    def compute_accuracy(
        self,
        similarity_matrix: tf.Tensor,
    ) -> tf.Tensor:
        """
        Top-1 retrieval accuracy.
        """

        labels = self.labels(
            tf.shape(
                similarity_matrix
            )[0]
        )

        prediction = tf.argmax(
            similarity_matrix,
            axis=1,
            output_type=tf.int32,
        )

        return tf.reduce_mean(

            tf.cast(

                tf.equal(
                    labels,
                    prediction,
                ),

                tf.float32,

            )

        )

    # ------------------------------------------------------

    def call(
        self,
        user_embedding: tf.Tensor,
        item_embedding: tf.Tensor,
    ):
        """
        Returns

        loss,
        metrics
        """

        matrix = self.similarity_matrix(
            user_embedding,
            item_embedding,
        )

        loss = self.compute_loss(
            matrix
        )

        accuracy = self.compute_accuracy(
            matrix
        )

        metrics = {

            "loss": loss,

            "accuracy": accuracy,

            "similarity": matrix,

        }

        return loss, metrics

    # ------------------------------------------------------

    def get_config(self):

        config = super().get_config()

        config.update(

            {

                "similarity":

                    tf.keras.utils.serialize_keras_object(
                        self.similarity
                    ),

            }

        )

        return config