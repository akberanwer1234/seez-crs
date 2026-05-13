"""
Multi‑agent recommender orchestrating multiple agent strategies.

Each agent implements a ``recommend(conversation) -> str`` method.
The orchestrator collects recommendations from all agents and returns
the movie title that appears most frequently. Ties are broken by
prioritising the recommendation from the first agent in the list.
"""

from __future__ import annotations

from collections import Counter
from typing import List


class MultiAgentRecommender:
    def __init__(self, agents: List[object]) -> None:
        """Initialise with a list of agents.

        Each agent must expose a synchronous ``recommend(conversation: str)``
        method returning a movie title string.
        """
        self.agents = agents

    async def recommend(self, question: str, history: List[str]) -> str:
        conversation = "\n".join(history + [question])
        # Collect recommendations from each agent
        preds = [agent.recommend(conversation) for agent in self.agents]
        if not preds:
            return "No recommendation found"
        # Majority vote
        counter = Counter(preds)
        most_common, count = counter.most_common(1)[0]
        return most_common
