from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Dict, List, Tuple

from ..datasets.movie_dataset import load_movie_dataset
from ..models.dynamic_fewshot import DynamicFewShotRecommender
from ..models.rag_enhanced import EnhancedRAGRecommender
from ..agents.enhanced_agent_system import PreferenceAwareAgentRecommender
from .metrics import compute_metrics


def split_dataset(examples: List[Dict[str, object]], test_ratio: float = 0.2, seed: int = 42) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Split a list of examples into train and test sets."""
    rng = random.Random(seed)
    shuffled = examples.copy()
    rng.shuffle(shuffled)
    n_test = int(len(shuffled) * test_ratio)
    test = shuffled[:n_test]
    train = shuffled[n_test:]
    return train, test


def evaluate_model(name: str, model, test_examples: List[Dict[str, object]], history_length: int = 3) -> Dict[str, float]:
    """Evaluate a recommendation model on a list of test examples."""
    metrics: Dict[str, List[float]] = {}
    for ex in test_examples:
        conversation_text: str = ex["text"]
        # For this evaluation we treat the entire conversation as the
        # ``question`` and pass an empty history.  A more realistic
        # evaluation could split turns and only provide the latest turn
        # along with previous context in ``history``.
        question = conversation_text
        history: List[str] = []
        truth_id: str = ex["rec_items"][0]
        # Obtain prediction from the model (synchronously for eval)
        # The models expose an async interface; we therefore run the
        # coroutine using ``asyncio.run``.  This avoids mixing event
        # loops in case evaluation is triggered from within an async
        # context.
        import asyncio  # local import to avoid global side effects
        if hasattr(model, "recommend_top_k"):
            pred_titles = asyncio.run(model.recommend_top_k(question, history, k=10))
        else:
            pred_title = asyncio.run(model.recommend(question, history))

            if isinstance(pred_title, tuple):
                pred_title = pred_title[0]

            pred_titles = [pred_title]

        pred_ids = []

        for title in pred_titles:
            if isinstance(title, tuple):
                title = title[0]

            pred_id = None

            if hasattr(model, "title_to_id"):
                pred_id = model.title_to_id.get(title)

            pred_ids.append(pred_id or title)

        # Remove duplicates while preserving ranking order.
        pred_ids = list(dict.fromkeys(pred_ids))
        metric_values = compute_metrics(pred_ids, truth_id, ks=[1, 5, 10])
        for key, value in metric_values.items():
            metrics.setdefault(key, []).append(value)
    # Average metrics over all test examples
    averaged = {k: statistics.mean(v) for k, v in metrics.items()}
    return averaged


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CRS models on LLM-REDIAL Movie dataset")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the Movie data directory containing Conversation.txt and final_data.jsonl")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of dataset used for testing (default 0.2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/test split")
    args = parser.parse_args()
    # Load dataset
    examples = load_movie_dataset(args.data_dir)
    train, test = split_dataset(examples, test_ratio=args.test_ratio, seed=args.seed)
    print(f"Loaded {len(examples)} examples; train={len(train)}, test={len(test)}")
    # Prepare item map and item frequencies for ranking and mapping
    item_map_path = Path(args.data_dir) / "item_map.json"
    item_map: Dict[str, str] = json.load(open(item_map_path, "r", encoding="utf-8"))
    # Build training data structures for models
    # dynamic few‑shot
    fewshot_model = DynamicFewShotRecommender(train, item_map=item_map, n_examples=5)
    # RAG
    rag_model = EnhancedRAGRecommender(train, item_map=item_map, top_k=20)
    # Agent
    agent_model = PreferenceAwareAgentRecommender(train, item_map=item_map, top_k=20)
    # Evaluate models
    results: Dict[str, Dict[str, float]] = {}
    for name, model in [
        ("dynamic_fewshot", fewshot_model),
        ("enhanced_rag", rag_model),
        ("preference_aware_agent", agent_model),
    ]:
        print(f"Evaluating {name}...", flush=True)
        scores = evaluate_model(name, model, test)
        results[name] = scores
    # Print summary table
    print("\nEvaluation summary (averaged over test set):")
    header = ["model", "mrr", "recall@1", "recall@5", "recall@10", "ndcg@1", "ndcg@5", "ndcg@10"]
    print("\t".join(header))
    for name, scores in results.items():
        print(f"{name}\t"
              f"{scores['mrr']:.3f}\t"
              f"{scores['recall@1']:.3f}\t"
              f"{scores['recall@5']:.3f}\t"
              f"{scores['recall@10']:.3f}\t"
              f"{scores['ndcg@1']:.3f}\t"
              f"{scores['ndcg@5']:.3f}\t"
              f"{scores['ndcg@10']:.3f}")


if __name__ == "__main__":
    main()