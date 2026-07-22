import numpy as np
import tensorflow as tf

from src.models.encoder import FeatureEncoder
from src.models.combiner import ConcatCombiner
from src.models.projection import MLPProjection
from src.models.tower import UserTower, ItemTower
from src.models.two_tower import TwoTowerModel

from src.serving import (
    BruteForceIndex,
    RetrievalPipeline,
)


def test_multi_feature_pipeline():

    # ==========================================================
    # Dummy vocabularies
    # ==========================================================

    user_vocab = ["1", "2", "3"]

    gender_vocab = [
        "M",
        "F",
    ]

    occupation_vocab = [
        "engineer",
        "teacher",
        "student",
    ]

    item_vocab = [
        "10",
        "20",
        "30",
        "40",
    ]

    genre_vocab = [
        "Action",
        "Drama",
        "Comedy",
    ]

    year_vocab = [
        "2020",
        "2021",
        "2022",
    ]

    # ==========================================================
    # User Tower
    # ==========================================================

    user_tower = UserTower(

        encoder=FeatureEncoder(

            {
                "user_id": user_vocab,
                "gender": gender_vocab,
                "occupation": occupation_vocab,
            },

            embedding_dim=16,

        ),

        combiner=ConcatCombiner(

            [
                "user_id",
                "gender",
                "occupation",
            ]

        ),

        projection=MLPProjection(

            output_dim=32,

        ),

    )

    # ==========================================================
    # Item Tower
    # ==========================================================

    item_tower = ItemTower(

        encoder=FeatureEncoder(

            {
                "movie_id": item_vocab,
                "genre": genre_vocab,
                "year": year_vocab,
            },

            embedding_dim=16,

        ),

        combiner=ConcatCombiner(

            [
                "movie_id",
                "genre",
                "year",
            ]

        ),

        projection=MLPProjection(

            output_dim=32,

        ),

    )

    model = TwoTowerModel(

        user_tower=user_tower,

        item_tower=item_tower,

    )

    # ==========================================================
    # Item Dataset
    # ==========================================================

    item_dataset = tf.data.Dataset.from_tensor_slices(

        {

            "movie_id": np.array(
                [
                    "10",
                    "20",
                    "30",
                    "40",
                ]
            ),

            "genre": np.array(
                [
                    "Action",
                    "Drama",
                    "Comedy",
                    "Action",
                ]
            ),

            "year": np.array(
                [
                    "2020",
                    "2021",
                    "2022",
                    "2020",
                ]
            ),

        }

    ).batch(2)

    # ==========================================================
    # Pipeline
    # ==========================================================

    pipeline = RetrievalPipeline(

        model=model,

        index=BruteForceIndex(),

    )

    pipeline.build(

        item_dataset,

        id_key="movie_id",

    )

    assert pipeline.index.size == 4

    # ==========================================================
    # User Query
    # ==========================================================

    user_features = {

        "user_id": tf.constant(["1"]),

        "gender": tf.constant(["M"]),

        "occupation": tf.constant(["engineer"]),

    }

    result = pipeline.recommend(

        user_features,

        top_k=3,

    )

    # ==========================================================
    # Assertions
    # ==========================================================

    assert len(result) == 3

    assert result.item_ids.shape == (3,)

    assert result.scores.shape == (3,)

    assert np.all(

        np.diff(result.scores) <= 0

    )

    print("=" * 60)

    print("Multi Feature Recommendation")

    print("=" * 60)

    for item_id, score in zip(

        result.item_ids,

        result.scores,

    ):

        if isinstance(item_id, (bytes, np.bytes_)):

            item_id = item_id.decode()

        print(

            f"{item_id:>6}  {score:.4f}"

        )