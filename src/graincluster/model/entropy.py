"""Entropy and objective computations.

Data term uses the Bayesian marginal likelihood (Dirichlet-Multinomial):

    L_data(C) = -log P(n | alpha, M)
              = log Γ(N + alpha*M) - log Γ(alpha*M)
                - sum_i [ log Γ(n_i + alpha) - log Γ(alpha) ]

This is the exact MDL "mixture code" length for a multinomial cluster with a
symmetric Dirichlet(alpha) prior over M joint (pair_type, bin) categories.
Zero-count categories drop out: Γ(0 + alpha) - Γ(alpha) = 0.

The prior concentration alpha controls prior strength: total pseudo-observations
= alpha * M. alpha → 0 recovers MLE; large alpha forces uniform.

Self-information for the frozen move-scoring model:
    I_C(t, b) = -log p̃_{t,b}(C),  p̃_i = (n_i + alpha) / (N + alpha*M)

This is exact: the marginal contribution of one new edge to L_data is
L_data(n + e_i) - L_data(n) = log(N + alpha*M) - log(n_i + alpha) = I_C(i).
So self_information() is consistent with the Bayesian data_term.
"""

from __future__ import annotations

import math
from math import lgamma

from .cluster import ClusterState


def cluster_entropy(
    cluster: ClusterState,
    M: int,
    alpha: float = 0.5,
) -> float:
    """Smoothed entropy H(p̃) for display/reporting only.

    Not used in the objective — L_data uses the Bayesian marginal.
    If N == 0, returns 0.
    """
    if cluster.N == 0:
        return 0.0

    N = cluster.N
    denom = N + alpha * M
    h = 0.0

    for count in cluster.counts.values():
        p = (count + alpha) / denom
        h -= p * math.log(p)

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
    """Bayesian marginal code length -log P(n | alpha, M) for one cluster.

    Zero-count categories contribute lgamma(alpha) - lgamma(alpha) = 0
    and are skipped automatically.
    """
    if cluster.N == 0:
        return 0.0
    N = cluster.N
    result = lgamma(N + alpha * M) - lgamma(alpha * M)
    result -= sum(lgamma(c + alpha) - lgamma(alpha) for c in cluster.counts.values())
    return result


def data_term_from_counts(
    counts: dict,
    N: int,
    M: int,
    alpha: float = 0.5,
) -> float:
    """Bayesian marginal code length from a raw counts dict + N.

    Used for exact delta scoring without a full ClusterState.
    """
    if N == 0:
        return 0.0
    result = lgamma(N + alpha * M) - lgamma(alpha * M)
    result -= sum(lgamma(c + alpha) - lgamma(alpha) for c in counts.values())
    return result


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
