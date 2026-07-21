"""
TensorFlow Dataset Builder

Converts pandas DataFrame into tf.data.Dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf

from src.data.preprocessing import ProcessedDataset

AUTOTUNE = tf.data.AUTOTUNE


@dataclass(slots=True)
class DatasetBundle:

    train: tf.data.Dataset

    test: tf.data.Dataset

    items: tf.data.Dataset

    user_vocab: list[str]

    item_vocab: list[str]

    movies: object


class DatasetBuilder:

    def __init__(
        self,
        batch_size: int = 512,
        shuffle_buffer: int = 100000,
    ):

        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    # -----------------------------------------------------

        # -----------------------------------------------------

    @staticmethod
    def item_dataframe_to_dataset(df):

        return tf.data.Dataset.from_tensor_slices(

            {

                "movie_id":
                    df["movieId"].astype(str).values,

                "title":
                    df["title"].values,

            }

        )

    @staticmethod
    def dataframe_to_dataset(df):

        return tf.data.Dataset.from_tensor_slices(

            {

                "user_id":
                    df["userId"].astype(str).values,

                "movie_id":
                    df["movieId"].astype(str).values,

                "rating":
                    df["rating"].astype("float32").values,

                "title":
                    df["title"].values,

            }

        )

    # -----------------------------------------------------

    def build(
        self,
        processed: ProcessedDataset,
    ) -> DatasetBundle:

        train_ds = self.dataframe_to_dataset(
            processed.train_df
        )

        test_ds = self.dataframe_to_dataset(
            processed.test_df
        )

        items_df = (

            processed.movies

            .drop_duplicates("movieId")

            .sort_values("movieId")

            .reset_index(drop=True)

        )

        item_ds = self.item_dataframe_to_dataset(
            items_df
        )

        item_ds = (

            item_ds

            .batch(self.batch_size)

            .prefetch(AUTOTUNE)

        )

        train_ds = (

            train_ds

            .shuffle(
                self.shuffle_buffer,
                reshuffle_each_iteration=True,
            )

            .batch(self.batch_size)

            .prefetch(AUTOTUNE)

        )

        test_ds = (

            test_ds

            .batch(self.batch_size)

            .prefetch(AUTOTUNE)

        )

        return DatasetBundle(

            train=train_ds,

            test=test_ds,

            items=item_ds,

            user_vocab=processed.user_vocab,

            item_vocab=processed.item_vocab,

            movies=processed.movies,

        )

if __name__ == "__main__":

    from src.data.preprocessing import MovieLensPreprocessor

    pre = MovieLensPreprocessor()

    processed = pre.process()

    builder = DatasetBuilder(batch_size=4)

    dataset = builder.build(processed)

    print(dataset)

    print()

    for batch in dataset.train.take(1):

        print(batch.keys())

        print()

        for k, v in batch.items():

            print(k)

            print(v.numpy())

            print()