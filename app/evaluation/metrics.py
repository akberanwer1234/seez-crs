"""Metrics for evaluating recommendation systems.

This module provides simple evaluation metrics commonly used in
information retrieval and recommender system research.  They include
Recall@k, Mean Reciprocal Rank (MRR) and Normalised Discounted
Cumulative Gain (NDCG).  These metrics assume a single ground‑truth
item per query (as is the case for the LLM‑REDIAL dataset where
``rec_item`` contains the target movie).  For multi‑label settings
they can be extended by modifying the relevant functions.

The functions take lists of predictions and ground truth and return
floating‑point scores.  They do not depend on any external libraries
and can be used both in offline evaluation scripts and within unit
tests.
"""

from __future__ import annotations

from math import log2
from typing import Iterable, List, Optional


def recall_at_k(pred: Iterable[str], truth: str, k: int) -> float:
    """Compute Recall@k for a single recommendation list.

    Recall@k is 1.0 if the ground truth item appears in the top ``k``
    positions of the prediction list, otherwise 0.0.

    Args:
        pred: Ordered list of predicted item identifiers.
        truth: The ground‑truth item identifier.
        k: Cut‑off rank.

    Returns:
        1.0 if ``truth`` is within the first ``k`` items in ``pred``,
        otherwise 0.0.
    """
    try:
        rank = list(pred).index(truth) + 1  # ranks are 1‑based
    except ValueError:
        return 0.0
    return 1.0 if rank <= k else 0.0


def reciprocal_rank(pred: Iterable[str], truth: str) -> float:
    """Compute the reciprocal rank for a single recommendation list.

    The reciprocal rank is the multiplicative inverse of the rank of
    the ground‑truth item.  If the truth does not appear in the
    predictions the score is 0.0.

    Args:
        pred: Ordered list of predicted item identifiers.
        truth: The ground‑truth item identifier.

    Returns:
        Reciprocal rank value between 0 and 1.
    """
    try:
        rank = list(pred).index(truth) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(pred: Iterable[str], truth: str, k: int) -> float:
    """Compute NDCG@k for a single recommendation list.

    For a single relevant item the Discounted Cumulative Gain (DCG)
    equals ``1 / log2(rank + 1)`` if the item is within the top ``k``
    predictions.  The ideal DCG (IDCG) is always ``1`` because
    positioning the relevant item at the top yields ``1 / log2(1 + 1)``.
    NDCG is therefore either ``1/log2(rank+1)`` if the item is in the
    list, or 0 if it is absent.

    Args:
        pred: Ordered list of predicted item identifiers.
        truth: The ground‑truth item identifier.
        k: Cut‑off rank.

    Returns:
        NDCG@k score between 0 and 1.
    """
    try:
        rank = list(pred).index(truth) + 1
    except ValueError:
        return 0.0
    if rank > k:
        return 0.0
    return 1.0 / log2(rank + 1)


def compute_metrics(pred: List[str], truth: str, ks: Optional[List[int]] = None) -> dict:
    """Compute a suite of metrics for a single query.

    Args:
        pred: Ordered list of predicted item identifiers.
        truth: The ground‑truth item identifier.
        ks: List of cut‑off values for which to compute Recall@k and
            NDCG@k.  If omitted defaults to [1, 5, 10].

    Returns:
        Dictionary containing ``mrr`` and, for each k in ``ks``,
        ``recall@k`` and ``ndcg@k``.
    """
    if ks is None:
        ks = [1, 5, 10]
    metrics = {
        'mrr': reciprocal_rank(pred, truth),
    }
    for k in ks:
        metrics[f'recall@{k}'] = recall_at_k(pred, truth, k)
        metrics[f'ndcg@{k}'] = ndcg_at_k(pred, truth, k)
    return metrics