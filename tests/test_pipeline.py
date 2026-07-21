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


def test_pipeline():

    model = DummyModel()

    dataset = [

        {

            "item_id": np.arange(100),

            "embedding": np.random.randn(
                100,
                32,
            ),

        }

    ]

    pipeline = RetrievalPipeline(

        model=model,

        index=BruteForceIndex(),

    )

    pipeline.build(
        dataset,
    )

    result = pipeline.recommend(

        {

            "embedding": dataset[0]["embedding"][0]

        },

        top_k=10,

    )

    assert len(result) == 10