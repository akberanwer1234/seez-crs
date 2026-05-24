from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Dict, List, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class _BaseLSARecommender:

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

    def __init__(self, train_examples: List[Dict[str, object]], item_map: Dict[str, str], n_examples: int = 5, n_components: int = 100) -> None:
        super().__init__(train_examples, item_map, n_components=n_components)
        self.n_examples = n_examples

    async def recommend(self, question: str, history: Optional[List[str]] = None) -> str:
        query_text = question
        # rec_id = await asyncio.to_thread(self._recommend_sync, query_text)
        loop = asyncio.get_event_loop()
        rec_id = await loop.run_in_executor(None, self._recommend_sync, query_text)
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

    def __init__(self, train_examples: List[Dict[str, object]], item_map: Dict[str, str], top_k: int = 20, n_components: int = 100) -> None:
        super().__init__(train_examples, item_map, n_components=n_components)
        self.top_k = top_k

    async def recommend(self, question: str, history: Optional[List[str]] = None) -> str:
        query_text = question
        # rec_id = await asyncio.to_thread(self._recommend_sync, query_text)
        loop = asyncio.get_event_loop()
        rec_id = await loop.run_in_executor(None, self._recommend_sync, query_text)
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