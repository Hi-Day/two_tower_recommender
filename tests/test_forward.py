"""
Forward Pass Integration Test

Run:

python -m tests.test_forward
"""

from __future__ import annotations

import tensorflow as tf

from src.data.preprocessing import MovieLensPreprocessor
from src.data.dataset import DatasetBuilder

from src.models.encoder import FeatureEncoder
from src.models.combiner import ConcatCombiner
from src.models.projection import MLPProjection
from src.models.tower import UserTower, ItemTower
from src.models.two_tower import TwoTowerModel


# ==========================================================
# Dataset
# ==========================================================

print("=" * 60)
print("Loading dataset...")
print("=" * 60)

processed = MovieLensPreprocessor().process()

bundle = DatasetBuilder(batch_size=4).build(processed)

batch = next(iter(bundle.train))

print("Batch Keys :", batch.keys())
print()

# ==========================================================
# User Tower
# ==========================================================

print("=" * 60)
print("Building User Tower")
print("=" * 60)

user_encoder = FeatureEncoder(
    vocabularies={
        "user_id": bundle.user_vocab,
    },
    embedding_dim=32,
)

user_combiner = ConcatCombiner(
    feature_names=[
        "user_id",
    ],
)

user_projection = MLPProjection(
    output_dim=32,
)

user_tower = UserTower(
    encoder=user_encoder,
    combiner=user_combiner,
    projection=user_projection,
)

# ==========================================================
# Item Tower
# ==========================================================

print("=" * 60)
print("Building Item Tower")
print("=" * 60)

item_encoder = FeatureEncoder(
    vocabularies={
        "movie_id": bundle.item_vocab,
    },
    embedding_dim=32,
)

item_combiner = ConcatCombiner(
    feature_names=[
        "movie_id",
    ],
)

item_projection = MLPProjection(
    output_dim=32,
)

item_tower = ItemTower(
    encoder=item_encoder,
    combiner=item_combiner,
    projection=item_projection,
)

# ==========================================================
# Two Tower
# ==========================================================

print("=" * 60)
print("Building TwoTowerModel")
print("=" * 60)

model = TwoTowerModel(
    user_tower=user_tower,
    item_tower=item_tower,
)

# ==========================================================
# Forward
# ==========================================================

print("=" * 60)
print("Running Forward Pass")
print("=" * 60)

user_features = {
    "user_id": batch["user_id"],
}

item_features = {
    "movie_id": batch["movie_id"],
}

output = model(
    {
        "user": user_features,
        "item": item_features,
    },
    training=False,
)

print()

print("=" * 60)
print("RESULT")
print("=" * 60)

print("User embedding shape :", output.user_embedding.shape)
print("Item embedding shape :", output.item_embedding.shape)

assert output.user_embedding.shape[0] == 4
assert output.item_embedding.shape[0] == 4

assert output.user_embedding.shape[1] == 32
assert output.item_embedding.shape[1] == 32

print()
print("Forward Pass SUCCESS")