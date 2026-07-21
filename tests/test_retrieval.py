"""
Test Retrieval Evaluation
"""

from src.data.preprocessing import MovieLensPreprocessor
from src.data.dataset import DatasetBuilder

from src.models.encoder import FeatureEncoder
from src.models.combiner import ConcatCombiner
from src.models.projection import MLPProjection
from src.models.tower import UserTower, ItemTower
from src.models.two_tower import TwoTowerModel

from src.evaluation.metrics import (
    RecallAtK,
    PrecisionAtK,
    HitRateAtK,
    MRR,
    NDCGAtK,
)

from src.evaluation.retrieval import RetrievalEvaluator

# =======================================================
# Dataset
# =======================================================

processed = MovieLensPreprocessor().process()

bundle = DatasetBuilder(
    batch_size=64,
).build(processed)

# =======================================================
# Build Model
# =======================================================

user_encoder = FeatureEncoder(
    vocabularies={
        "user_id": bundle.user_vocab,
    },
    embedding_dim=32,
)

item_encoder = FeatureEncoder(
    vocabularies={
        "movie_id": bundle.item_vocab,
    },
    embedding_dim=32,
)

user_tower = UserTower(
    encoder=user_encoder,
    combiner=ConcatCombiner(["user_id"]),
    projection=MLPProjection(output_dim=32),
)

item_tower = ItemTower(
    encoder=item_encoder,
    combiner=ConcatCombiner(["movie_id"]),
    projection=MLPProjection(output_dim=32),
)

model = TwoTowerModel(
    user_tower=user_tower,
    item_tower=item_tower,
)

# =======================================================
# Item Dataset
# =======================================================

item_dataset = (
    bundle.train
    .map(lambda x: {
        "movie_id": x["movie_id"]
    })
)

# =======================================================
# Evaluator
# =======================================================

evaluator = RetrievalEvaluator(
    model=model,
    item_dataset=item_dataset,
)

metrics = [
    RecallAtK(10),
    RecallAtK(20),
    PrecisionAtK(10),
    HitRateAtK(10),
    MRR(),
    NDCGAtK(10),
]

results = evaluator.evaluate(
    bundle.test,
    metrics,
)

print()

print("=" * 60)

for name, value in results.items():
    print(f"{name:20s}: {value:.4f}")

print("=" * 60)