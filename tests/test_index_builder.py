import numpy as np

from src.serving import (
    BruteForceIndex,
    IndexBuilder,
)


class DummyEncoder:

    def encode_item(self, batch):
        return batch["embedding"]


def test_build_index():

    dataset = [
        {
            "item_id": np.array([1, 2]),
            "embedding": np.random.randn(2, 32),
        }
    ]

    builder = IndexBuilder(
        DummyEncoder(),
    )

    index = BruteForceIndex()

    builder.build(
        dataset,
        index,
    )

    assert index.size == 2