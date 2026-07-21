"""
Training Step Integration Test

Run

python -m tests.test_training_step
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
from src.models.retrieval import (
    DotProductSimilarity,
    RetrievalTask,
)

# ==========================================================
# Dataset
# ==========================================================

processed = MovieLensPreprocessor().process()

bundle = DatasetBuilder(
    batch_size=8,
).build(processed)

batch = next(iter(bundle.train))

# ==========================================================
# User Tower
# ==========================================================

user_encoder = FeatureEncoder(
    vocabularies={
        "user_id": bundle.user_vocab,
    },
    embedding_dim=32,
)

user_combiner = ConcatCombiner(
    feature_names=["user_id"],
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

item_encoder = FeatureEncoder(
    vocabularies={
        "movie_id": bundle.item_vocab,
    },
    embedding_dim=32,
)

item_combiner = ConcatCombiner(
    feature_names=["movie_id"],
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
# Model
# ==========================================================

model = TwoTowerModel(
    user_tower=user_tower,
    item_tower=item_tower,
)

retrieval = RetrievalTask(
    similarity=DotProductSimilarity(),
)

optimizer = tf.keras.optimizers.Adam(1e-3)

# ==========================================================
# Features
# ==========================================================

user_features = {
    "user_id": batch["user_id"],
}

item_features = {
    "movie_id": batch["movie_id"],
}

# ==========================================================
# Training Step
# ==========================================================

print("=" * 60)
print("Running training step...")
print("=" * 60)

with tf.GradientTape() as tape:

    output = model(
        {
            "user": user_features,
            "item": item_features,
        },
        training=True,
    )

    loss, metrics = retrieval(
        output.user_embedding,
        output.item_embedding,
    )

variables = model.trainable_variables

gradients = tape.gradient(
    loss,
    variables,
)

# ==========================================================
# Validate Gradients
# ==========================================================

num_none = sum(g is None for g in gradients)

print(f"Trainable Variables : {len(variables)}")
print(f"None Gradients      : {num_none}")

assert num_none == 0, "Some gradients are None!"

# ==========================================================
# Optimizer Step
# ==========================================================

optimizer.apply_gradients(
    zip(
        gradients,
        variables,
    )
)

# ==========================================================
# Result
# ==========================================================

print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print("Loss      :", float(loss))
print("Accuracy  :", float(metrics["accuracy"]))
print("Variables :", len(variables))