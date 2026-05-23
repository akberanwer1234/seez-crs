"""Semantic recommendation using Latent Semantic Analysis (LSA).

This module implements lightweight semantic recommenders based on
Latent Semantic Analysis (LSA).  LSA reduces the dimensionality of
TF‑IDF representations via truncated SVD, capturing latent concepts
beyond surface word overlap.  Similarity in the reduced space can
better reflect semantic similarity between conversations, improving
recommendation quality while remaining computationally efficient.

Two recommender classes are provided:

* ``LSAFewShotRecommender`` – analogous to the dynamic few‑shot model
  but operating in the LSA space.  It retrieves a small number of
  nearest neighbours and aggregates their recommended items.

* ``LSARAGRecommender`` – analogous to the enhanced RAG model but
  retrieving a larger pool of neighbours in the LSA space.

Both classes use ``TruncatedSVD`` from scikit‑learn.  The number of
latent dimensions can be configured; 100 is a reasonable default for
thousands of conversations.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class _BaseLSARecommender:
    """Shared logic for LSA recommenders.

    Args:
        train_examples: Training dataset containing at least ``'text'`` and
            ``'rec_items'`` keys.
        item_map: Mapping from item IDs to titles.
        n_components: Number of latent dimensions for SVD.
    """

    def __init__(self, train_examples: List[Dict[str, object]], item_map: Dict[str, str], n_components: int = 100) -> None:
        self.item_map = item_map
        self.train_texts: List[str] = []
        self.train_rec_items: List[List[str]] = []
        for ex in train_examples:
            text = ex.get("text")
            rec_items = ex.get("rec_items")
            if isinstance(text, str) and text.strip() and rec_items:
                self.train_texts.append(text.strip())
                self.train_rec_items.append(list(rec_items))
        # Lexical TF‑IDF
        self.vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = self.vectorizer.fit_transform(self.train_texts)
        # LSA via truncated SVD
        # n_components should be less than min(n_samples, n_features)
        n_comp = min(n_components, tfidf_matrix.shape[0] - 1, tfidf_matrix.shape[1] - 1)
        self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
        self.train_lsa = self.svd.fit_transform(tfidf_matrix)
        # Precompute item frequency for popularity weighting
        self.item_freq: Dict[str, int] = {}
        for recs in self.train_rec_items:
            for iid in recs:
                self.item_freq[iid] = self.item_freq.get(iid, 0) + 1
        # Title→ID mapping
        self.title_to_id: Dict[str, str] = {v: k for k, v in item_map.items()}

    def _score_candidates(self, sims: List[float], top_indices: List[int]) -> Dict[str, float]:
        """Aggregate candidate scores based on similarities and popularity.

        Args:
            sims: List of similarity scores for all training examples.
            top_indices: Indices of retrieved conversations to consider.

        Returns:
            Dictionary mapping item IDs to their aggregated scores.
        """
        candidate_scores: Dict[str, float] = defaultdict(float)
        for idx in top_indices:
            similarity = sims[idx]
            if similarity <= 0:
                continue
            rec_items = self.train_rec_items[idx]
            for item_id in rec_items:
                pop = self.item_freq.get(item_id, 1)
                candidate_scores[item_id] += similarity * (1 + 0.05 * pop)
        return candidate_scores


class LSAFewShotRecommender(_BaseLSARecommender):
    """Dynamic few‑shot model using latent semantic analysis.

    Retrieves a handful of nearest neighbours in the LSA space and
    aggregates their recommended items.  Suitable for quick responses
    where exact semantics are important.
    """

    def __init__(self, train_examples: List[Dict[str, object]], item_map: Dict[str, str], n_examples: int = 5, n_components: int = 100) -> None:
        super().__init__(train_examples, item_map, n_components=n_components)
        self.n_examples = n_examples

    async def recommend(self, question: str, history: Optional[List[str]] = None) -> str:
        query_text = question
        rec_id = await asyncio.to_thread(self._recommend_sync, query_text)
        return self.item_map.get(rec_id, rec_id)

    def _recommend_sync(self, query_text: str) -> str:
        # Compute LSA vector for query
        tfidf_q = self.vectorizer.transform([query_text])
        lsa_q = self.svd.transform(tfidf_q)
        sims = cosine_similarity(lsa_q, self.train_lsa).flatten()
        k = min(self.n_examples, len(sims))
        top_indices = sims.argsort()[::-1][:k]
        candidate_scores = self._score_candidates(sims, top_indices)
        if not candidate_scores:
            return "No recommendation found"
        best_item = max(candidate_scores.items(), key=lambda x: x[1])[0]
        return best_item


class LSARAGRecommender(_BaseLSARecommender):
    """Enhanced RAG model using latent semantic analysis.

    Retrieves a larger set of neighbours in the LSA space to build a
    broad evidence pool.  Aggregates candidate item scores and
    returns the highest‑scoring movie.
    """

    def __init__(self, train_examples: List[Dict[str, object]], item_map: Dict[str, str], top_k: int = 20, n_components: int = 100) -> None:
        super().__init__(train_examples, item_map, n_components=n_components)
        self.top_k = top_k

    async def recommend(self, question: str, history: Optional[List[str]] = None) -> str:
        query_text = question
        rec_id = await asyncio.to_thread(self._recommend_sync, query_text)
        return self.item_map.get(rec_id, rec_id)

    def _recommend_sync(self, query_text: str) -> str:
        tfidf_q = self.vectorizer.transform([query_text])
        lsa_q = self.svd.transform(tfidf_q)
        sims = cosine_similarity(lsa_q, self.train_lsa).flatten()
        k = min(self.top_k, len(sims))
        top_indices = sims.argsort()[::-1][:k]
        candidate_scores = self._score_candidates(sims, top_indices)
        if not candidate_scores:
            return "No recommendation found"
        best_item = max(candidate_scores.items(), key=lambda x: x[1])[0]
        return best_item