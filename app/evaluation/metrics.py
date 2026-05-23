from __future__ import annotations

from math import log2
from typing import Iterable, List, Optional


def recall_at_k(pred: Iterable[str], truth: str, k: int) -> float:
    try:
        rank = list(pred).index(truth) + 1  # ranks are 1‑based
    except ValueError:
        return 0.0
    return 1.0 if rank <= k else 0.0


def reciprocal_rank(pred: Iterable[str], truth: str) -> float:
    try:
        rank = list(pred).index(truth) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(pred: Iterable[str], truth: str, k: int) -> float:
    try:
        rank = list(pred).index(truth) + 1
    except ValueError:
        return 0.0
    if rank > k:
        return 0.0
    return 1.0 / log2(rank + 1)


def compute_metrics(pred: List[str], truth: str, ks: Optional[List[int]] = None) -> dict:
    if ks is None:
        ks = [1, 5, 10]
    metrics = {
        'mrr': reciprocal_rank(pred, truth),
    }
    for k in ks:
        metrics[f'recall@{k}'] = recall_at_k(pred, truth, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(pred, truth, k)
    return metrics