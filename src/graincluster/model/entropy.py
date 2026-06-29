"""Entropy and objective computations.

All entropy uses Dirichlet smoothing (pseudocount alpha > 0) to avoid log(0).
M = total joint categories = sum of n_bins per pair type.

Smoothed probability:
    p̃_i(C) = (n_i(C) + alpha) / (N_C + alpha * M)

Joint entropy:
    H_C = -sum_i p̃_i * log(p̃_i)

Data term contribution of cluster C:
    L_data(C) = N_C * H_C

Self-information of category i under cluster C (frozen model):
    I_C(i) = -log(p̃_i(C))
"""

from __future__ import annotations

import math

from .cluster import ClusterState


def _smoothed_prob(count: int, total: int, alpha: float, M: int) -> float:
    return (count + alpha) / (total + alpha * M)


def cluster_entropy(
    cluster: ClusterState,
    M: int,
    alpha: float = 0.5,
) -> float:
    """Joint entropy H_C with Dirichlet smoothing over M categories.

    If N == 0 (no internal edges), entropy is 0 by convention.
    """
    if cluster.N == 0:
        return 0.0

    N = cluster.N
    denom = N + alpha * M
    h = 0.0

    for count in cluster.counts.values():
        p = (count + alpha) / denom
        h -= p * math.log(p)

    # categories with count == 0 contribute (alpha / denom) * log(alpha / denom)
    # number of zero-count categories = M - len(cluster.counts)
    n_zero = M - len(cluster.counts)
    if n_zero > 0 and alpha > 0.0:
        p_zero = alpha / denom
        h -= n_zero * p_zero * math.log(p_zero)

    return h


def data_term(
    cluster: ClusterState,
    M: int,
    alpha: float = 0.5,
) -> float:
    """N_C * H_C for one cluster."""
    return cluster.N * cluster_entropy(cluster, M, alpha)


def data_term_from_counts(
    counts: dict,
    N: int,
    M: int,
    alpha: float = 0.5,
) -> float:
    """data_term (N*H) computed from a raw counts dict + N.

    Used for exact delta scoring without a full ClusterState.
    """
    if N == 0:
        return 0.0
    denom = N + alpha * M
    h = 0.0
    for count in counts.values():
        p = (count + alpha) / denom
        h -= p * math.log(p)
    n_zero = M - len(counts)
    if n_zero > 0 and alpha > 0.0:
        p_zero = alpha / denom
        h -= n_zero * p_zero * math.log(p_zero)
    return N * h


def self_information(
    pair_type_idx: int,
    bin_idx: int,
    cluster: ClusterState,
    M: int,
    alpha: float = 0.5,
) -> float:
    """I_C(t, b) = -log(p̃_{t,b}(C)).

    Used for frozen-model move scoring. The cluster state is NOT mutated.
    """
    key = (pair_type_idx, bin_idx)
    count = cluster.counts.get(key, 0)
    N = cluster.N
    p = (count + alpha) / (N + alpha * M)
    return -math.log(p)
