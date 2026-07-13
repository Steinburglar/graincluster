"""Parameterized scoring for cluster identity models.

This module scores multinomial blocks using explicit cluster-specific
parameters instead of marginalizing them away. The default estimate is a
floor-constrained Dirichlet MAP mode. Posterior mean remains available for
comparison and diagnostics.

The current multinomial block score is an asymptotic two-part code:

- exact discrete count code under ``theta_hat``
- asymptotic parameter code ``-log p(theta_hat) + (d/2) log N``

Exact multinomial and Dirichlet normalizers are evaluated cheaply via
``math.lgamma``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .cluster import ClusterState

EstimatorName = str


def dirichlet_alpha_from_base(base: np.ndarray, kappa: float) -> np.ndarray:
    """Return alpha = kappa * normalized base."""
    base = np.asarray(base, dtype=float)
    if base.ndim != 1:
        raise ValueError("Dirichlet base must be one-dimensional")
    if np.any(base < 0.0):
        raise ValueError("Dirichlet base cannot contain negative values")
    total = float(base.sum())
    if total <= 0.0:
        raise ValueError("Dirichlet base must have positive sum")
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    return kappa * (base / total)


def uniform_alpha(n_categories: int, kappa: float) -> np.ndarray:
    """Uniform-base Dirichlet alpha with total concentration kappa."""
    if n_categories <= 0:
        raise ValueError("n_categories must be positive")
    return np.full(n_categories, kappa / n_categories, dtype=float)


def cluster_count_prior_predictive_cost(
    k_clusters: int,
    mean: float,
    strength: float,
) -> float:
    """Negative log prior predictive for K under a Poisson-Gamma model.

    Uses:

    - ``K | lambda ~ Poisson(lambda)``
    - ``lambda ~ Gamma(a=strength, b=strength/mean)`` with rate ``b``

    so that ``E[lambda] = mean`` and the scalar ``strength`` controls how
    tightly lambda is concentrated around that mean.
    """
    if k_clusters < 0:
        raise ValueError("k_clusters must be nonnegative")
    if mean <= 0.0:
        raise ValueError("mean must be positive")
    if strength <= 0.0:
        raise ValueError("strength must be positive")

    rate = strength / mean
    k = float(k_clusters)
    a = float(strength)
    return (
        math.lgamma(a)
        + math.lgamma(k + 1.0)
        - math.lgamma(k + a)
        - a * math.log(rate)
        + (k + a) * math.log(rate + 1.0)
    )


def lomax_cut_prior_cost(
    cut_value: float,
    n_atoms: int,
    k_clusters: int,
    beta0: float,
) -> float:
    """Negative log prior predictive for total cut under an Exponential-Exponential model.

    The hierarchy is:

    - ``x | lambda, N, K ~ Exponential(lambda)``
    - ``lambda | N, K ~ Exponential(beta_NK)``

    with

    ```text
    beta_NK = beta0 * N^(2/3) * K^(1/3)
    ```

    This is the ``k = 1`` special case of the more general Exponential-Gamma
    mixture, yielding a Lomax prior predictive with fixed shape 1.
    """
    if cut_value < 0.0:
        raise ValueError("cut_value must be nonnegative")
    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive")
    if k_clusters < 0:
        raise ValueError("k_clusters must be nonnegative")
    if beta0 <= 0.0:
        raise ValueError("beta0 must be positive")

    k_eff = max(1.0, float(k_clusters))
    beta_nk = float(beta0) * (float(n_atoms) ** (2.0 / 3.0)) * (k_eff ** (1.0 / 3.0))
    return math.log(beta_nk) + 2.0 * math.log1p(float(cut_value) / beta_nk)


def posterior_mean(counts: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Posterior-mean multinomial parameter for counts and Dirichlet alpha."""
    counts = np.asarray(counts, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if counts.shape != alpha.shape:
        raise ValueError("counts and alpha must have the same shape")
    if np.any(counts < 0.0):
        raise ValueError("counts cannot contain negative values")
    if np.any(alpha <= 0.0):
        raise ValueError("alpha must be strictly positive")
    denom = float(counts.sum() + alpha.sum())
    if denom <= 0.0:
        raise ValueError("posterior mean denominator must be positive")
    return (counts + alpha) / denom


def constrained_map(
    counts: np.ndarray,
    alpha: np.ndarray,
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Floor-constrained Dirichlet MAP estimate.

    Maximizes sum_i (n_i + alpha_i - 1) log(theta_i) with theta_i >= epsilon
    and sum theta_i = 1 using the standard active-set form.  Categories with
    nonpositive effective weight are placed at the floor; positive-weight
    categories share the remaining probability mass in proportion to their
    weights.  If no category has positive weight, the least-bad category gets
    the remaining mass.
    """
    counts = np.asarray(counts, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if counts.shape != alpha.shape:
        raise ValueError("counts and alpha must have the same shape")
    if np.any(counts < 0.0):
        raise ValueError("counts cannot contain negative values")
    if np.any(alpha <= 0.0):
        raise ValueError("alpha must be strictly positive")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    if epsilon * len(counts) >= 1.0:
        raise ValueError("epsilon too large for category count")

    weights = counts + alpha - 1.0
    theta = np.full(len(counts), epsilon, dtype=float)
    active = weights > 0.0
    remaining = 1.0 - epsilon * len(counts)

    if np.any(active):
        theta[active] += remaining * weights[active] / float(weights[active].sum())
    else:
        theta[int(np.argmax(weights))] += remaining

    return theta / float(theta.sum())


def estimate_theta(
    counts: np.ndarray,
    alpha: np.ndarray,
    estimator: EstimatorName = "constrained_map",
    epsilon: float = 1e-12,
) -> np.ndarray:
    """Estimate multinomial theta by configured estimator."""
    if estimator == "constrained_map":
        return constrained_map(counts, alpha, epsilon=epsilon)
    if estimator == "posterior_mean":
        return posterior_mean(counts, alpha)
    raise ValueError(f"Unknown parameter estimator: {estimator}")


def parameterized_multinomial_cost(
    counts: np.ndarray,
    alpha: np.ndarray,
    estimator: EstimatorName = "constrained_map",
    epsilon: float = 1e-12,
) -> float:
    """Approximate two-part code for multinomial counts with Dirichlet prior.

    The current implementation uses:

    - a discrete code length for the observed count vector under ``theta_hat``
    - an asymptotic parameter code

    ```text
    L(theta_hat) ~= -log p(theta_hat) + (d / 2) log N
    ```

    where ``d = K - 1`` is the number of free multinomial parameters and
    ``N`` is the total count for this block.
    """
    counts = np.asarray(counts, dtype=float)
    alpha = np.asarray(alpha, dtype=float)
    if counts.shape != alpha.shape:
        raise ValueError("counts and alpha must have the same shape")
    if np.any(counts < 0.0):
        raise ValueError("counts cannot contain negative values")
    if np.any(alpha <= 0.0):
        raise ValueError("alpha must be strictly positive")

    total = float(counts.sum())
    if total <= 0.0:
        return 0.0

    theta = estimate_theta(counts, alpha, estimator=estimator, epsilon=epsilon)

    # multinomial coefficient: log(N!) - sum_i log(n_i!)
    count_constant = math.lgamma(total + 1.0) - float(
        np.sum([math.lgamma(float(n) + 1.0) for n in counts])
    )

    # Dirichlet normalizer: log B(alpha) = sum_i log Gamma(alpha_i) - log Gamma(alpha_0)
    alpha_sum = float(alpha.sum())
    dirichlet_normalizer = float(np.sum([math.lgamma(float(a)) for a in alpha])) - math.lgamma(alpha_sum)

    data_cost = -float(np.dot(counts, np.log(theta)))
    prior_density_cost = -float(np.dot(alpha - 1.0, np.log(theta)))

    n_categories = len(alpha)
    d_free = max(0, n_categories - 1)
    parameter_resolution_cost = 0.5 * d_free * math.log(total)

    return (
        -count_constant
        + data_cost
        + dirichlet_normalizer
        + prior_density_cost
        + parameter_resolution_cost
    )


def dense_counts_from_mapping(
    counts: Mapping[int, int],
    n_categories: int,
) -> np.ndarray:
    """Convert sparse integer counts to dense category vector."""
    dense = np.zeros(n_categories, dtype=float)
    for idx, count in counts.items():
        if idx < 0 or idx >= n_categories:
            raise ValueError(f"category index {idx} outside 0..{n_categories - 1}")
        dense[idx] = count
    return dense


def species_data_term(
    species_counts: Mapping[int, int],
    alpha_species: np.ndarray,
    estimator: EstimatorName = "constrained_map",
    epsilon: float = 1e-12,
) -> float:
    """Parameterized species-composition cost for one cluster."""
    counts = dense_counts_from_mapping(species_counts, len(alpha_species))
    if counts.sum() == 0:
        return 0.0
    return parameterized_multinomial_cost(
        counts, alpha_species, estimator=estimator, epsilon=epsilon
    )


def edge_data_term_for_pair_type(
    bin_counts: Mapping[int, int],
    alpha_edge: np.ndarray,
    estimator: EstimatorName = "constrained_map",
    epsilon: float = 1e-12,
) -> float:
    """Parameterized edge-bin cost for one pair type in one cluster."""
    counts = dense_counts_from_mapping(bin_counts, len(alpha_edge))
    if counts.sum() == 0:
        return 0.0
    return parameterized_multinomial_cost(
        counts, alpha_edge, estimator=estimator, epsilon=epsilon
    )


def cluster_data_term(
    cluster: ClusterState,
    alpha_species: np.ndarray,
    alpha_edge_by_type: Mapping[int, np.ndarray],
    estimator: EstimatorName = "constrained_map",
    epsilon: float = 1e-12,
) -> float:
    """Species plus pair-type-conditional edge-bin cost for one cluster."""
    result = species_data_term(
        cluster.species_counts,
        alpha_species,
        estimator=estimator,
        epsilon=epsilon,
    )
    for pair_type_idx, alpha_edge in alpha_edge_by_type.items():
        bin_counts = cluster.edge_counts_for_pair_type(pair_type_idx)
        result += edge_data_term_for_pair_type(
            bin_counts,
            alpha_edge,
            estimator=estimator,
            epsilon=epsilon,
        )
    return result


def category_self_information(
    category_idx: int,
    counts: Mapping[int, int],
    alpha: np.ndarray,
    estimator: EstimatorName = "constrained_map",
    epsilon: float = 1e-12,
) -> float:
    """-log estimated probability for one category."""
    if category_idx < 0 or category_idx >= len(alpha):
        raise ValueError(f"category index {category_idx} outside alpha range")
    dense = dense_counts_from_mapping(counts, len(alpha))
    theta = estimate_theta(dense, alpha, estimator=estimator, epsilon=epsilon)
    return -math.log(float(theta[category_idx]))
