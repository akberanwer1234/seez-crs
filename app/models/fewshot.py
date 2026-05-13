"""
Few‑shot recommendation based on TF–IDF similarity.

This recommender selects a small number of example dialogues from the
dataset at initialisation time. To produce a recommendation, it
vectorises the current conversation (history + question) using a
TF–IDF vectoriser trained on the examples and chooses the example with
the highest cosine similarity. The associated movie title from that
example is returned as the recommendation.
"""

from __future__ import annotations

from typing import List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class FewShotRecommender:
    def __init__(self, examples: List[Tuple[str, List[str]]], num_examples: int = 3) -> None:
        """Initialise the few‑shot recommender.

        Args:
            examples: A list of ``(text, titles)`` where ``text`` is a
                conversation transcript and ``titles`` is a list of human‑
                readable recommended movie names for that conversation.
            num_examples: Number of examples to retain from ``examples``.
        """
        self.examples = examples[: num_examples]
        self.conversations = [ex[0] for ex in self.examples]
        self.rec_titles = [ex[1] for ex in self.examples]
        # Fit TF–IDF on the example conversations
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.example_vectors = self.vectorizer.fit_transform(self.conversations)

    async def recommend(self, question: str, history: List[str]) -> str:
        """Recommend a movie given the dialogue history and new question.

        Args:
            question: The latest user utterance.
            history: Previous conversation utterances.

        Returns:
            The first recommended movie title from the most similar example.
        """
        query_text = "\n".join(history + [question])
        query_vector = self.vectorizer.transform([query_text])
        sims = cosine_similarity(query_vector, self.example_vectors).flatten()
        best_idx = int(sims.argmax())
        titles = self.rec_titles[best_idx]
        return titles[0] if titles else "No recommendation found"
