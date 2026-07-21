from src.data.preprocessing import MovieLensPreprocessor
from src.data.dataset import DatasetBuilder

from src.models.encoder import FeatureEncoder
from src.models.combiner import ConcatCombiner
from src.models.projection import MLPProjection
from src.models.tower import UserTower, ItemTower
from src.models.two_tower import TwoTowerModel

from src.serving import (
    BruteForceIndex,
    RetrievalPipeline,
)

import numpy as np

def test_movielens_pipeline():
    # ==========================================================
    # Dataset
    # ==========================================================

    processed = MovieLensPreprocessor().process()

    bundle = DatasetBuilder(
        batch_size=64,
    ).build(processed)

    # ==========================================================
    # Model
    # ==========================================================

    user_tower = UserTower(
        encoder=FeatureEncoder(
            {"user_id": bundle.user_vocab},
            embedding_dim=32,
        ),
        combiner=ConcatCombiner(["user_id"]),
        projection=MLPProjection(output_dim=32),
    )

    item_tower = ItemTower(
        encoder=FeatureEncoder(
            {"movie_id": bundle.item_vocab},
            embedding_dim=32,
        ),
        combiner=ConcatCombiner(["movie_id"]),
        projection=MLPProjection(output_dim=32),
    )

    model = TwoTowerModel(
        user_tower=user_tower,
        item_tower=item_tower,
    )

    pipeline = RetrievalPipeline(
        model=model,
        index=BruteForceIndex(),
    )

    pipeline.build(
        bundle.items,
        id_key="movie_id",
    )

    assert pipeline.index.size > 0

    batch = next(iter(bundle.test))

    user_features = {
        "user_id": batch["user_id"][:1],
    }

    result = pipeline.recommend(
        user_features,
        top_k=10,
    )

    assert len(result) == 10

    assert result.item_ids.shape == (10,)

    assert result.scores.shape == (10,)

    print("=" * 60)
    print("Top Recommendations")
    print("=" * 60)

    for item_id, score in zip(result.item_ids, result.scores):

        if isinstance(item_id, (bytes, np.bytes_)):
            item_id = item_id.decode("utf-8")

        print(f"{item_id:>8} {score:.4f}")


    assert np.all(np.diff(result.scores) <= 0)

    assert len(np.unique(result.item_ids)) == len(result.item_ids)