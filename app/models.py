

from __future__ import annotations

import asyncio
from collections import Counter
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaseRecommender:
    async def recommend(self, question: str, history: List[str]) -> str:
        raise NotImplementedError


class FewShotRecommender(BaseRecommender):

    def __init__(self, examples: List[Tuple[str, List[str]]], num_examples: int = 3):
        # Retain the specified number of examples
        self.examples = examples[: num_examples]
        self.conversations = [ex[0] for ex in self.examples]
        # Fit a TF–IDF vectoriser on the example conversations
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.example_vectors = self.vectorizer.fit_transform(self.conversations)
        self.rec_titles = [ex[1] for ex in self.examples]

    async def recommend(self, question: str, history: List[str]) -> str:
        # Build the query from history and question
        query_text = "\n".join(history + [question])
        query_vector = self.vectorizer.transform([query_text])
        # Compute cosine similarity to each example
        sims = cosine_similarity(query_vector, self.example_vectors).flatten()
        # Choose the most similar example
        best_idx = int(sims.argmax())
        recommended = self.rec_titles[best_idx][0] if self.rec_titles[best_idx] else "No recommendation found"
        return recommended


class RAGRecommender(BaseRecommender):

    def __init__(self, examples: List[Tuple[str, List[str]]], top_k: int = 3):
        self.examples = examples
        self.conversations = [ex[0] for ex in self.examples]
        self.rec_titles_list = [ex[1] for ex in self.examples]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.example_vectors = self.vectorizer.fit_transform(self.conversations)
        self.top_k = top_k

    async def recommend(self, question: str, history: List[str]) -> str:
        query_text = "\n".join(history + [question])
        query_vector = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vector, self.example_vectors).flatten()
        # Get indices of top‑k similar conversations
        top_indices = sims.argsort()[::-1][: self.top_k]
        # Aggregate recommended titles across retrieved examples
        candidate_titles = []
        for idx in top_indices:
            titles = self.rec_titles_list[idx]
            candidate_titles.extend(titles)
        if not candidate_titles:
            return "No recommendation found"
        # Choose the most common title among candidates
        most_common, _ = Counter(candidate_titles).most_common(1)[0]
        return most_common
