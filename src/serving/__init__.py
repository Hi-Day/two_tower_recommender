"""
Serving Layer
=============

High-level retrieval and recommendation interfaces.
"""

from .builder import IndexBuilder
from .engine import RecommendationEngine
from .pipeline import RetrievalPipeline

from .index import BaseIndex
from .brute_force import BruteForceIndex

from .result import RecommendationResult

__all__ = [
    "BaseIndex",
    "BruteForceIndex",
    "RecommendationResult",
    "IndexBuilder",
    "RecommendationEngine",
    "RetrievalPipeline",
    "SearchResult"
]