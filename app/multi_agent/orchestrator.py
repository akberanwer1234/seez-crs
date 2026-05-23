from __future__ import annotations

from collections import Counter
from typing import List


class MultiAgentRecommender:
    def __init__(self, agents: List[object]) -> None:
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
