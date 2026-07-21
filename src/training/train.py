"""
Training Entry Point
====================

TensorFlow 2.19
Keras 3
"""

from __future__ import annotations

import tensorflow as tf

from src.data.preprocessing import MovieLensPreprocessor
from src.data.dataset import DatasetBuilder

from src.models.encoder import FeatureEncoder
from src.models.combiner import ConcatCombiner
from src.models.projection import MLPProjection
from src.models.tower import (
    UserTower,
    ItemTower,
)
from src.models.two_tower import TwoTowerModel

from src.models.retrieval import (
    DotProductSimilarity,
    RetrievalTask,
)

from src.training.trainer import Trainer


# ==========================================================
# Hyperparameters
# ==========================================================

EMBEDDING_DIM = 64
PROJECTION_DIM = 64

BATCH_SIZE = 256
EPOCHS = 10

# RETRIEVAL
TEMPERATURE = 0.5

LEARNING_RATE = 1e-3

CHECKPOINT = "checkpoints/two_tower.weights.h5"

# ==========================================================
# Dataset
# ==========================================================

print("=" * 60)
print("Loading Dataset")
print("=" * 60)

processed = MovieLensPreprocessor().process()

bundle = DatasetBuilder(
    batch_size=BATCH_SIZE,
).build(processed)


# ==========================================================
# User Tower
# ==========================================================

print("Building User Tower...")

user_encoder = FeatureEncoder(
    vocabularies={
        "user_id": bundle.user_vocab,
    },
    embedding_dim=EMBEDDING_DIM,
)

user_combiner = ConcatCombiner(
    feature_names=[
        "user_id",
    ],
)

user_projection = MLPProjection(
    output_dim=PROJECTION_DIM,
)

user_tower = UserTower(
    encoder=user_encoder,
    combiner=user_combiner,
    projection=user_projection,
)


# ==========================================================
# Item Tower
# ==========================================================

print("Building Item Tower...")

item_encoder = FeatureEncoder(
    vocabularies={
        "movie_id": bundle.item_vocab,
    },
    embedding_dim=EMBEDDING_DIM,
)

item_combiner = ConcatCombiner(
    feature_names=[
        "movie_id",
    ],
)

item_projection = MLPProjection(
    output_dim=PROJECTION_DIM,
)

item_tower = ItemTower(
    encoder=item_encoder,
    combiner=item_combiner,
    projection=item_projection,
)


# ==========================================================
# Model
# ==========================================================

print("Building Two-Tower Model...")

model = TwoTowerModel(
    user_tower=user_tower,
    item_tower=item_tower,
)

retrieval_task = RetrievalTask(
    similarity=DotProductSimilarity(
        temperature=TEMPERATURE,
    ),
)

optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
)

optimizer = tf.keras.optimizers.AdamW(
    learning_rate=1e-3,
    weight_decay=1e-5,
)

# ==========================================================
# Trainer
# ==========================================================

trainer = Trainer(
    model=model,
    retrieval_task=retrieval_task,
    optimizer=optimizer,
    callbacks=[

        EarlyStopping(
            monitor="val_loss",
            patience=3,
        ),

        ModelCheckpoint(
            "checkpoints/best.weights.h5",
        ),

    ],
)


# ==========================================================
# Training
# ==========================================================

trainer.fit(
    train_dataset=bundle.train,
    validation_dataset=bundle.test,
    epochs=EPOCHS,
)


# ==========================================================
# Save Checkpoint
# ==========================================================

trainer.save(
    CHECKPOINT,
)


# ==========================================================
# Evaluation
# ==========================================================

if hasattr(bundle, "items"):

    trainer.evaluate(
        test_dataset=bundle.test,
        item_dataset=bundle.items,
    )

else:

    print()
    print("=" * 60)
    print("Evaluation skipped")
    print("=" * 60)
    print(
        "DatasetBuilder belum menyediakan bundle.items."
    )


print()
print("=" * 60)
print("Done.")
print("=" * 60)