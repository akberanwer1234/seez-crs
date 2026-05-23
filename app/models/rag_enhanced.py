from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EnhancedRAGRecommender:

    def __init__(self, train_examples: List[Dict[str, object]], item_map: Dict[str, str], top_k: int = 20) -> None:
        self.item_map = item_map
        self.top_k = top_k
        # Extract training texts and associated rec items
        self.train_texts: List[str] = []
        self.train_rec_items: List[List[str]] = []
        for ex in train_examples:
            text = ex.get("text")
            rec_items = ex.get("rec_items")
            if isinstance(text, str) and text.strip() and rec_items:
                self.train_texts.append(text.strip())
                self.train_rec_items.append(list(rec_items))
        # Vectoriser and matrix
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.train_vectors = self.vectorizer.fit_transform(self.train_texts)
        # Item frequencies
        self.item_freq: Dict[str, int] = {}
        for recs in self.train_rec_items:
            for iid in recs:
                self.item_freq[iid] = self.item_freq.get(iid, 0) + 1
        # Title→ID mapping
        self.title_to_id: Dict[str, str] = {v: k for k, v in item_map.items()}

    # async def recommend(self, question: str, history: Optional[List[str]] = None) -> str:
    #     query_text = question

    #     loop = asyncio.get_event_loop()
    #     rec_id = await loop.run_in_executor(
    #         None,
    #         self._recommend_sync,
    #         query_text
    #     )

    #     return self.item_map.get(rec_id, rec_id)

    async def recommend(self, question: str, history: Optional[List[str]] = None) -> str:
        titles = await self.recommend_top_k(question, history, k=1)
        return titles[0] if titles else "No recommendation found"


    async def recommend_top_k(
        self,
        question: str,
        history: Optional[List[str]] = None,
        k: int = 10,
    ) -> List[str]:
        query_text = question

        loop = asyncio.get_running_loop()
        rec_ids = await loop.run_in_executor(
            None,
            self._recommend_top_k_sync,
            query_text,
            k,
        )

        return [self.item_map.get(rec_id, rec_id) for rec_id in rec_ids]


    def _recommend_top_k_sync(self, query_text: str, k: int = 10) -> List[str]:
        """Return ranked top-k item identifiers."""

        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.train_vectors).flatten()

        retrieval_k = min(self.top_k, len(sims))
        top_indices = sims.argsort()[::-1][:retrieval_k]

        candidate_scores: Dict[str, float] = defaultdict(float)

        for idx in top_indices:
            similarity = sims[idx]

            if similarity <= 0:
                continue

            rec_items = self.train_rec_items[idx]

            for item_id in rec_items:
                popularity = self.item_freq.get(item_id, 1)
                candidate_scores[item_id] += similarity * (1 + 0.05 * popularity)

        if not candidate_scores:
            return []

        ranked_items = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [item_id for item_id, score in ranked_items[:k]]

    def _recommend_sync(self, query_text: str) -> str:
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.train_vectors).flatten()
        # Determine retrieval size
        k = min(self.top_k, len(sims))
        top_indices = sims.argsort()[::-1][:k]
        candidate_scores: Dict[str, float] = defaultdict(float)
        for idx in top_indices:
            similarity = sims[idx]
            if similarity <= 0:
                continue
            rec_items = self.train_rec_items[idx]
            for item_id in rec_items:
                popularity = self.item_freq.get(item_id, 1)
                from math import log1p
                # candidate_scores[item_id] += similarity * (1 + 0.05 * popularity)
                candidate_scores[item_id] += similarity * (1 + 0.1 * log1p(popularity))
        if not candidate_scores:
            return "No recommendation found"
        best_item = max(candidate_scores.items(), key=lambda x: x[1])[0]
        return best_item
    