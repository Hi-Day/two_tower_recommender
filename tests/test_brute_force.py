import numpy as np
import pytest

from src.serving import BruteForceIndex


@pytest.fixture
def embeddings():

    rng = np.random.default_rng(42)

    emb = rng.normal(size=(100, 32)).astype(np.float32)

    emb /= np.linalg.norm(
        emb,
        axis=1,
        keepdims=True,
    )

    return emb


@pytest.fixture
def item_ids():

    return np.arange(100)


@pytest.fixture
def index(
    embeddings,
    item_ids,
):

    index = BruteForceIndex()

    index.build(
        embeddings,
        item_ids,
    )

    return index


def test_build(index):

    assert index.size == 100

    assert index.dimension == 32


def test_search_returns_top_k(
    index,
    embeddings,
):

    result = index.search(
        embeddings[0],
        top_k=10,
    )

    assert len(result) == 10


def test_first_result_is_itself(
    index,
    embeddings,
):

    result = index.search(
        embeddings[0],
        top_k=5,
    )

    assert result.item_ids[0] == 0


def test_scores_sorted(
    index,
    embeddings,
):

    result = index.search(
        embeddings[0],
        top_k=20,
    )

    assert np.all(
        np.diff(result.scores) <= 0
    )


def test_top_k_larger_than_index(
    index,
    embeddings,
):

    result = index.search(
        embeddings[0],
        top_k=500,
    )

    assert len(result) == 100


def test_len(index):

    assert len(index) == 100


def test_repr(index):

    assert "BruteForceIndex" in repr(index)


def test_empty_index():

    index = BruteForceIndex()

    with pytest.raises(RuntimeError):

        index.search(
            np.random.randn(32),
        )