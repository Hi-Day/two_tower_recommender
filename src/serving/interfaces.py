"""
Serving Interfaces
==================
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class UserEncoder(Protocol):

    def encode_user(
        self,
        features,
    ) -> np.ndarray:
        ...


class ItemEncoder(Protocol):

    def encode_item(
        self,
        features,
    ) -> np.ndarray:
        ...