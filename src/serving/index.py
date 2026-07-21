from abc import ABC
from abc import abstractmethod

import numpy as np

from .search_result import SearchResult


class BaseIndex(ABC):

    @abstractmethod
    def build(
        self,
        embeddings: np.ndarray,
        item_ids: np.ndarray,
    ) -> None:
        ...

    @abstractmethod
    def search(
        self,
        query: np.ndarray,
        top_k: int,
    ) -> SearchResult:
        ...

    @abstractmethod
    def save(
        self,
        path: str,
    ) -> None:
        ...

    @abstractmethod
    def load(
        self,
        path: str,
    ) -> None:
        ...