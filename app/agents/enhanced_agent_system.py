from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PreferenceAwareAgentRecommender:
    """Preference‑aware conversational recommender.

    Args:
        train_examples: List of examples for training.  Each example
            must contain ``'text'`` and ``'rec_items'``.
        item_map: Mapping from item identifiers to movie titles.
        top_k: Number of similar conversations to retrieve.
    """

    def __init__(self, train_examples: List[Dict[str, object]], item_map: Dict[str, str], top_k: int = 20) -> None:
        self.item_map = item_map
        self.top_k = top_k
        # Prepare training data
        self.train_texts: List[str] = []
        self.train_rec_items: List[List[str]] = []
        for ex in train_examples:
            text = ex.get("text")
            rec_items = ex.get("rec_items")
            if isinstance(text, str) and text.strip() and rec_items:
                self.train_texts.append(text.strip())
                self.train_rec_items.append(list(rec_items))
        # TF‑IDF
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.train_vectors = self.vectorizer.fit_transform(self.train_texts)
        # Item frequency
        self.item_freq: Dict[str, int] = {}
        for recs in self.train_rec_items:
            for iid in recs:
                self.item_freq[iid] = self.item_freq.get(iid, 0) + 1
        # Title to ID mapping
        self.title_to_id: Dict[str, str] = {v: k for k, v in item_map.items()}
        # Precompute lowercase titles for quick substring search
        self.lower_title_map: Dict[str, str] = {v.lower(): v for v in item_map.values()}

    # async def recommend(self, question: str, history: Optional[List[str]] = None) -> Tuple[str, str]:
    #     query_text = question

    #     loop = asyncio.get_event_loop()
    #     rec_id, explanation = await loop.run_in_executor(
    #         None,
    #         self._recommend_sync,
    #         query_text
    #     )

    #     title = self.item_map.get(rec_id, rec_id)
    #     return title, explanation

    async def recommend(self, question: str, history: Optional[List[str]] = None) -> Tuple[str, str]:
        query_text = question

        loop = asyncio.get_running_loop()
        rec_ids, liked_titles = await loop.run_in_executor(
            None,
            self._recommend_top_k_sync,
            query_text,
            1,
        )

        if not rec_ids:
            return "No recommendation found", "Sorry, I couldn't find a suitable recommendation."

        best_item = rec_ids[0]
        rec_title = self.item_map.get(best_item, best_item)

        if liked_titles:
            liked_str = ", ".join(liked_titles)
            explanation = f"Because you mentioned liking {liked_str}, you might also enjoy {rec_title}."
        else:
            explanation = f"I recommend {rec_title} based on similar conversations."

        return rec_title, explanation


    async def recommend_top_k(
        self,
        question: str,
        history: Optional[List[str]] = None,
        k: int = 10,
    ) -> List[str]:
        query_text = question

        loop = asyncio.get_running_loop()
        rec_ids, _ = await loop.run_in_executor(
            None,
            self._recommend_top_k_sync,
            query_text,
            k,
        )

        return [self.item_map.get(rec_id, rec_id) for rec_id in rec_ids]


    def _recommend_top_k_sync(self, query_text: str, k: int = 10) -> Tuple[List[str], List[str]]:
        """Return ranked top-k item identifiers and extracted liked titles."""

        liked_titles: List[str] = []
        lower_query = query_text.lower()

        for title in self.item_map.values():
            words = title.split()

            if len(words) <= 5 and title.lower() in lower_query:
                liked_titles.append(title)

        liked_ids = [
            self.title_to_id.get(title)
            for title in liked_titles
            if self.title_to_id.get(title)
        ]

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
                if item_id in liked_ids:
                    continue

                popularity = self.item_freq.get(item_id, 1)
                score = similarity * (1 + 0.05 * popularity)

                title = self.item_map.get(item_id, "")
                pref_bonus = 0.0

                for liked in liked_titles:
                    liked_tokens = set(liked.lower().split())
                    item_tokens = set(title.lower().split())
                    overlap = liked_tokens & item_tokens

                    if overlap:
                        pref_bonus += 0.2 * len(overlap)

                candidate_scores[item_id] += score + pref_bonus

        if not candidate_scores:
            return [], liked_titles

        ranked_items = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        return [item_id for item_id, score in ranked_items[:k]], liked_titles

    def _recommend_sync(self, query_text: str) -> Tuple[str, str]:
        # Extract liked movies by scanning for known titles in the query
        liked_titles: List[str] = []
        lower_query = query_text.lower()
        # naive substring matching of movie titles (lowercase); we
        # restrict to titles less than 5 words to reduce false positives
        for title in self.item_map.values():
            words = title.split()
            if len(words) <= 5 and title.lower() in lower_query:
                liked_titles.append(title)
        liked_ids = [self.title_to_id.get(t) for t in liked_titles if self.title_to_id.get(t)]
        # Compute query vector and similarities
        query_vec = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vec, self.train_vectors).flatten()
        k = min(self.top_k, len(sims))
        top_indices = sims.argsort()[::-1][:k]
        candidate_scores: Dict[str, float] = defaultdict(float)
        for idx in top_indices:
            similarity = sims[idx]
            if similarity <= 0:
                continue
            rec_items = self.train_rec_items[idx]
            for item_id in rec_items:
                # Skip items explicitly mentioned by the user (avoid recommending same movie)
                if item_id in liked_ids:
                    continue
                popularity = self.item_freq.get(item_id, 1)
                from math import log1p
                # score = similarity * (1 + 0.05 * popularity)
                score = similarity * (1 + 0.1 * log1p(popularity))
                # Preference match: boost items sharing words with liked titles
                title = self.item_map.get(item_id, "")
                pref_bonus = 0.0
                for liked in liked_titles:
                    # simple token overlap bonus
                    liked_tokens = set(liked.lower().split())
                    item_tokens = set(title.lower().split())
                    overlap = liked_tokens & item_tokens
                    if overlap:
                        pref_bonus += 0.2 * len(overlap)
                candidate_scores[item_id] += score + pref_bonus
        if not candidate_scores:
            return "No recommendation found", "Sorry, I couldn't find a suitable recommendation."
        best_item = max(candidate_scores.items(), key=lambda x: x[1])[0]
        # Build explanation
        rec_title = self.item_map.get(best_item, best_item)
        if liked_titles:
            liked_str = ", ".join(liked_titles)
            explanation = f"Because you mentioned liking {liked_str}, you might also enjoy {rec_title}."
        else:
            explanation = f"I recommend {rec_title} based on similar conversations."
        return best_item, explanation