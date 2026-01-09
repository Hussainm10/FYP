"""
Lightweight cross-encoder reranker for query–candidate pairs.

- Runs on CPU by default
- Small model for good speed: cross-encoder/ms-marco-MiniLM-L-6-v2
- API: score_pairs(query, candidates) -> list[float]
"""

from __future__ import annotations
from typing import Iterable, List, Tuple

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    CPU-friendly reranker that scores (query, text) pairs.
    """

    def __init__(
            self,
            model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device: str = "cpu",
    ) -> None:
        """
        Args:
            model_name: HuggingFace cross-encoder model id.
            device: "cpu" or "cuda".
        """
        self.model = CrossEncoder(model_name, device=device)

    def score_pairs(self, query: str, texts: Iterable[str]) -> List[float]:
        """
        Score many (query, text) pairs.

        Args:
            query: User query string.
            texts: Candidate passages to be scored with the query.

        Returns:
            List of float scores aligned with `texts`.
        """
        pairs: List[Tuple[str, str]] = [(query, t) for t in texts]
        if not pairs:
            return []
        scores: List[float] = self.model.predict(pairs).tolist()
        return scores
