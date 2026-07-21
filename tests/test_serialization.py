import numpy as np

from src.serving import BruteForceIndex


def test_save_load(
    tmp_path,
):

    embeddings = np.random.randn(
        100,
        32,
    )

    ids = np.arange(100)

    index = BruteForceIndex()

    index.build(
        embeddings,
        ids,
    )

    path = tmp_path / "index.npz"

    index.save(path)

    loaded = BruteForceIndex()

    loaded.load(path)

    assert loaded.size == index.size

    assert np.allclose(

        loaded.embeddings,

        index.embeddings,

    )

    assert np.all(

        loaded.item_ids == index.item_ids

    )