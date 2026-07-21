"""
Global configuration for the Two-Tower Recommendation System.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Project directories
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"

RAW_DATA_DIR = DATA_DIR / "raw"

PROCESSED_DATA_DIR = DATA_DIR / "processed"

INDEX_DIR = DATA_DIR / "index"

MODEL_DIR = ROOT_DIR / "models"

EMBEDDING_DIR = ROOT_DIR / "embeddings"

LOG_DIR = ROOT_DIR / "logs"

# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

DATASET_URL = (
    "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
)

DATASET_NAME = "ml-latest-small"

# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

EMBED_DIM = 128

HIDDEN_DIM = 256

BATCH_SIZE = 512

EPOCHS = 10

LEARNING_RATE = 1e-3

SEED = 42

# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

TOP_K = [5, 10, 20, 50]

# ---------------------------------------------------------------------
# Create directories automatically
# ---------------------------------------------------------------------

for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    INDEX_DIR,
    MODEL_DIR,
    EMBEDDING_DIR,
    LOG_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)