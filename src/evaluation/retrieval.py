from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import tensorflow as tf

from .metrics import Metric


# ==========================================================
# Retrieval Result
# ==========================================================

@dataclass(slots=True)
class RetrievalResult:
    """
    Result returned by RetrievalEvaluator.
    """

    predictions: np.ndarray
    targets: np.ndarray
    scores: np.ndarray | None = None


# ==========================================================
# Retrieval Evaluator
# ==========================================================

class RetrievalEvaluator:

    def __init__(
        self,
        model,
        item_dataset: tf.data.Dataset,
    ):
        self.model = model
        self.item_dataset = item_dataset

        self.item_ids = None
        self.item_embeddings = None

    # ======================================================
    # Item Index
    # ======================================================

    def build_item_index(self):

        item_ids = []
        embeddings = []

        for batch in self.item_dataset:

            embedding = self.model.encode_item(batch)

            embeddings.append(embedding.numpy())

            item_ids.extend(batch["movie_id"].numpy())

        self.item_ids = np.asarray(item_ids)

        self.item_embeddings = np.concatenate(
            embeddings,
            axis=0,
        )

        return self

    # ======================================================
    # Retrieve
    # ======================================================

    def retrieve(
        self,
        user_dataset,
        top_k: int = 10,
    ) -> RetrievalResult:

        if self.item_embeddings is None:
            self.build_item_index()

        predictions = []
        targets = []
        scores_all = []

        item_matrix = tf.convert_to_tensor(
            self.item_embeddings,
            dtype=tf.float32,
        )

        for batch in user_dataset:

            user_embedding = self.model.encode_user(batch)

            scores = tf.matmul(
                user_embedding,
                item_matrix,
                transpose_b=True,
            )

            values, indices = tf.math.top_k(
                scores,
                k=top_k,
            )

            #
            # IMPORTANT
            #
            # Convert embedding index
            # back to original movie_id
            #
            movie_ids = self.item_ids[
                indices.numpy()
            ]

            predictions.append(movie_ids)

            scores_all.append(
                values.numpy()
            )

            targets.extend(
                batch["movie_id"].numpy()
            )

        return RetrievalResult(
            predictions=np.concatenate(
                predictions,
                axis=0,
            ),
            targets=np.asarray(targets),
            scores=np.concatenate(
                scores_all,
                axis=0,
            ),
        )

    # ======================================================
    # Evaluate
    # ======================================================

    def evaluate(
        self,
        user_dataset,
        metrics: Iterable[Metric],
        top_k: int | None = None,
    ):

        if top_k is None:

            top_k = max(
                getattr(metric, "k", 1)
                for metric in metrics
            )

        result = self.retrieve(
            user_dataset,
            top_k=top_k,
        )

        outputs = {}

        for metric in metrics:

            outputs[
                metric.name
            ] = metric(
                result.predictions,
                result.targets,
            )

        return outputs

    # ======================================================
    # Config
    # ======================================================

    def get_config(self):

        return {
            "num_items":
                None
                if self.item_ids is None
                else len(self.item_ids),
        }

    def __repr__(self):

        if self.item_ids is None:
            return "RetrievalEvaluator(unbuilt)"

        return (
            f"RetrievalEvaluator("
            f"num_items={len(self.item_ids)})"
        )