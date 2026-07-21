"""
Retrieval Pipeline
==================
"""

from __future__ import annotations

from .builder import IndexBuilder
from .engine import RecommendationEngine


class RetrievalPipeline:

    def __init__(
        self,
        model,
        index,
    ):

        self.index = index

        self.indexer = IndexBuilder(model)

        self.engine = RecommendationEngine(
            model,
            index,
        )

    def build(
        self,
        dataset,
        id_key="item_id",
    ):

        self.indexer.build(
            dataset,
            self.index,
            id_key=id_key,
        )

        return self

    # -------------------------------------------------

    def save(
        self,
        path,
    ):

        self.index.save(path)

    # -------------------------------------------------

    def load(
        self,
        path,
    ):

        self.index.load(path)

        return self

    # -------------------------------------------------

    def recommend(
        self,
        features,
        top_k=10,
    ):

        return self.engine.recommend(
            features,
            top_k,
        )