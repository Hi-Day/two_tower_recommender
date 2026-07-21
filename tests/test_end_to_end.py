import numpy as np

from src.serving import (
    BruteForceIndex,
    RetrievalPipeline,
)


class DummyModel:

    def encode_item(
        self,
        batch,
    ):

        return batch["embedding"]

    def encode_user(
        self,
        batch,
    ):

        return batch["embedding"]


def test_end_to_end():

    rng = np.random.default_rng(42)

    embeddings = rng.normal(
        size=(200, 32),
    )

    dataset = [

        {

            "item_id": np.arange(200),

            "embedding": embeddings,

        }

    ]

    pipeline = RetrievalPipeline(

        model=DummyModel(),

        index=BruteForceIndex(),

    )

    pipeline.build(
        dataset,
    )

    result = pipeline.recommend(

        {

            "embedding": embeddings[0]

        },

        top_k=20,

    )

    assert len(result) == 20

    assert result.item_ids[0] == 0

    assert np.all(

        np.diff(result.scores) <= 0

    )