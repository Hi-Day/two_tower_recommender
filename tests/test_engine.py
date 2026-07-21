import numpy as np

from src.serving import (
    BruteForceIndex,
    RecommendationEngine,
)


class DummyUserEncoder:

    def encode_user(
        self,
        features,
    ):

        return features["embedding"]


def test_recommend():

    embeddings = np.random.randn(
        50,
        32,
    )

    ids = np.arange(50)

    index = BruteForceIndex()

    index.build(
        embeddings,
        ids,
    )

    engine = RecommendationEngine(

        DummyUserEncoder(),

        index,

    )

    result = engine.recommend(

        {

            "embedding": embeddings[0]

        },

        top_k=10,

    )

    assert len(result) == 10

    assert result.item_ids[0] == 0