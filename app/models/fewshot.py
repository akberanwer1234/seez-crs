from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FewShotRecommender:
    def __init__(self, examples: List[Tuple[str, List[str]]], num_examples: int = 3) -> None:
        self.examples = examples[: num_examples]
        self.conversations = [ex[0] for ex in self.examples]
        self.rec_titles = [ex[1] for ex in self.examples]
        # Fit TF–IDF on the example conversations
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.example_vectors = self.vectorizer.fit_transform(self.conversations)

    async def recommend(self, question: str, history: List[str]) -> str:
        query_text = "\n".join(history + [question])
        query_vector = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vector, self.example_vectors).flatten()
        best_idx = int(sims.argmax())
        titles = self.rec_titles[best_idx]
        return titles[0] if titles else "No recommendation found"
