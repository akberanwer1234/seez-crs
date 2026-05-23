from __future__ import annotations

import random
from collections import Counter
from typing import Dict, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityAgent:
    """Agent recommending the most similar movie title to the conversation."""

    def __init__(self, item_titles: List[str]) -> None:
        self.item_titles = item_titles
        # Fit TF–IDF on the movie titles
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.item_vectors = self.vectorizer.fit_transform(self.item_titles)

    def recommend(self, conversation: str) -> str:
        query_vec = self.vectorizer.transform([conversation])
        sims = cosine_similarity(query_vec, self.item_vectors).flatten()
        best_idx = int(sims.argmax())
        return self.item_titles[best_idx]


class PopularityAgent:
    """Agent recommending the globally most popular movie."""

    def __init__(self, item_frequencies: Dict[str, int], item_map: Dict[str, str]) -> None:
        self.item_frequencies = item_frequencies
        self.item_map = item_map
        # Determine the most popular item ID based on frequency
        if item_frequencies:
            self.most_popular_id, _ = Counter(item_frequencies).most_common(1)[0]
        else:
            self.most_popular_id = None

    def recommend(self, conversation: str) -> str:
        if self.most_popular_id is None:
            return "No recommendation found"
        # Return the human‑readable title
        return self.item_map.get(self.most_popular_id, self.most_popular_id)


class RandomAgent:
    """Agent recommending a random movie title."""

    def __init__(self, item_titles: List[str]) -> None:
        self.item_titles = item_titles

    def recommend(self, conversation: str) -> str:
        if not self.item_titles:
            return "No recommendation found"
        return random.choice(self.item_titles)


class AgentBasedRecommender:
    """High‑level recommender that delegates to a selected agent"""

    def __init__(self, similarity_agent: SimilarityAgent) -> None:
        self.similarity_agent = similarity_agent

    async def recommend(self, question: str, history: List[str]) -> str:
        conversation = "\n".join(history + [question])
        return self.similarity_agent.recommend(conversation)
