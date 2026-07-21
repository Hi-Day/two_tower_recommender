"""
Data preprocessing module for Two-Tower Recommendation System.

Responsibilities
----------------
1. Load MovieLens dataset
2. Clean data
3. Merge ratings & movies
4. Encode IDs
5. Build vocabularies
6. Train/Test split
"""

from pathlib import Path
from dataclasses import dataclass
import pandas as pd

from sklearn.model_selection import train_test_split

from src.config import (
    RAW_DATA_DIR,
    DATASET_NAME,
    SEED,
)

@dataclass(slots=True)
class ProcessedDataset:
    """
    Container hasil preprocessing.
    """

    train_df: pd.DataFrame
    test_df: pd.DataFrame

    user_vocab: list[str]
    item_vocab: list[str]

    movies: pd.DataFrame

class MovieLensPreprocessor:

    def __init__(self):

        dataset = RAW_DATA_DIR / DATASET_NAME

        self.ratings_path = dataset / "ratings.csv"

        self.movies_path = dataset / "movies.csv"

    # --------------------------------------------------

    def load(self):

        ratings = pd.read_csv(self.ratings_path)

        movies = pd.read_csv(self.movies_path)

        return ratings, movies

    # --------------------------------------------------

    def merge(self):

        ratings, movies = self.load()

        df = ratings.merge(
            movies,
            on="movieId",
            how="left"
        )

        return df

    # --------------------------------------------------

    def clean(self, df):

        df = df.drop_duplicates()

        df = df.dropna()

        return df

    # --------------------------------------------------

    def build_vocab(self, df):

        user_vocab = sorted(
            df["userId"]
            .astype(str)
            .unique()
            .tolist()
        )

        item_vocab = sorted(
            df["movieId"]
            .astype(str)
            .unique()
            .tolist()
        )

        return user_vocab, item_vocab

    # --------------------------------------------------

    def split(self, df):

        train_df, test_df = train_test_split(

            df,

            test_size=0.2,

            random_state=SEED,

            shuffle=True

        )

        return train_df, test_df

    # --------------------------------------------------

    def process(self) -> ProcessedDataset:

        df = self.merge()

        df = self.clean(df)

        user_vocab, item_vocab = self.build_vocab(df)

        train_df, test_df = self.split(df)

        movies = (
            df[["movieId", "title"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        return ProcessedDataset(
            train_df=train_df,
            test_df=test_df,
            user_vocab=user_vocab,
            item_vocab=item_vocab,
            movies=movies,
        )


# ----------------------------------------------------------

if __name__ == "__main__":

    preprocessor = MovieLensPreprocessor()

    processed = preprocessor.process()

    print(f"Train : {len(processed.train_df)}")
    print(f"Test  : {len(processed.test_df)}")

    print(f"Users : {len(processed.user_vocab)}")
    print(f"Movies: {len(processed.item_vocab)}")