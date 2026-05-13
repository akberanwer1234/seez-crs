"""Model package exposing different recommendation strategies."""

from .fewshot import FewShotRecommender
from .rag import RAGRecommender

__all__ = ["FewShotRecommender", "RAGRecommender"]