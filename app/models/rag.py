from __future__ import annotations

from collections import Counter
from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGRecommender:
    def __init__(self, examples: List[Tuple[str, List[str]]], top_k: int = 3) -> None:
        self.examples = examples
        self.conversations = [ex[0] for ex in self.examples]
        self.rec_titles_list = [ex[1] for ex in self.examples]
        self.top_k = top_k
        # Fit TF–IDF on all conversations
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.example_vectors = self.vectorizer.fit_transform(self.conversations)

    async def recommend(self, question: str, history: List[str]) -> str:
        query_text = "\n".join(history + [question])
        query_vector = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vector, self.example_vectors).flatten()
        # Indices of top‑k most similar conversations
        top_indices = sims.argsort()[::-1][: self.top_k]
        candidate_titles = []
        for idx in top_indices:
            titles = self.rec_titles_list[idx]
            candidate_titles.extend(titles)
        if not candidate_titles:
            return "No recommendation found"
        most_common, _ = Counter(candidate_titles).most_common(1)[0]
        return most_common
