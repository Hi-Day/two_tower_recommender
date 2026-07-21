"""
Index Builder
=============

Build retrieval indices from item encoders.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf

from .index import BaseIndex
from .interfaces import ItemEncoder

from tqdm.auto import tqdm


class IndexBuilder:
    """
    Build an index from an item encoder.
    """

    def __init__(
        self,
        encoder: ItemEncoder,
    ):
        self.encoder = encoder

    # -----------------------------------------------------

    def build(
        self,
        dataset,
        index: BaseIndex,
        id_key: str = "item_id",
    ) -> BaseIndex:

        embeddings = []
        item_ids = []

        for batch in tqdm(
            dataset,
            desc="Encoding Items",
        ):

            embedding = self.encoder.encode_item(
                batch
            )

            if isinstance(
                embedding,
                tf.Tensor,
            ):
                embedding = embedding.numpy()

            embeddings.append(embedding)

            ids = batch[id_key]

            if isinstance(ids, tf.Tensor):
                ids = ids.numpy()

            item_ids.append(np.asarray(ids))

        embeddings = np.concatenate(
            embeddings,
            axis=0,
        )

        item_ids = np.concatenate(
            item_ids,
            axis=0,
        )

        index.build(
            embeddings,
            item_ids,
        )

        print(

            f"Indexed {len(item_ids)} items "

            f"({embeddings.shape[1]} dimensions)"

        )

        return index