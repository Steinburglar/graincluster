"""Tests for parameterized MAP scoring utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from graincluster.model.parameterized import (
    cluster_count_prior_predictive_cost,
    constrained_map,
    dirichlet_alpha_from_base,
    lomax_cut_prior_cost,
    parameterized_multinomial_cost,
    posterior_mean,
    uniform_alpha,
)


class TestDirichletUtilities:
    def test_uniform_alpha_sums_to_kappa(self):
        alpha = uniform_alpha(5, kappa=2.5)
        assert alpha.sum() == pytest.approx(2.5)
        assert np.all(alpha == pytest.approx(0.5))

    def test_dirichlet_alpha_from_base_normalizes_base(self):
        alpha = dirichlet_alpha_from_base(np.array([2.0, 1.0]), kappa=3.0)
        assert alpha.tolist() == pytest.approx([2.0, 1.0])

    def test_cluster_count_prior_predictive_prefers_meanish_k(self):
        near = cluster_count_prior_predictive_cost(k_clusters=10, mean=10.0, strength=4.0)
        far = cluster_count_prior_predictive_cost(k_clusters=40, mean=10.0, strength=4.0)
        assert near < far

    def test_lomax_cut_prior_prefers_small_cut(self):
        small = lomax_cut_prior_cost(cut_value=1.0, n_atoms=1000, k_clusters=2, beta0=1.0)
        large = lomax_cut_prior_cost(cut_value=100.0, n_atoms=1000, k_clusters=2, beta0=1.0)
        assert small < large

    def test_posterior_mean_keeps_unseen_categories_positive(self):
        counts = np.array([10.0, 0.0])
        alpha = np.array([0.5, 0.5])
        theta = posterior_mean(counts, alpha)
        assert theta[1] > 0.0
        assert theta.sum() == pytest.approx(1.0)

    def test_constrained_map_keeps_unseen_categories_at_floor(self):
        counts = np.array([10.0, 0.0])
        alpha = np.array([0.5, 0.5])
        theta = constrained_map(counts, alpha, epsilon=1e-6)
        assert theta[1] == pytest.approx(1e-6)
        assert theta.sum() == pytest.approx(1.0)

    def test_constrained_map_matches_interior_map(self):
        counts = np.array([10.0, 5.0])
        alpha = np.array([3.0, 4.0])
        theta = constrained_map(counts, alpha)
        expected = (counts + alpha - 1.0) / (counts.sum() + alpha.sum() - 2)
        assert theta == pytest.approx(expected)


class TestPriorShape:
    def test_parameterized_cost_matches_full_joint_formula(self):
        counts = np.array([3.0, 2.0])
        alpha = np.array([0.5, 0.5])
        theta = constrained_map(counts, alpha, epsilon=1e-12)
        count_constant = math.lgamma(counts.sum() + 1.0) - sum(
            math.lgamma(float(n) + 1.0) for n in counts
        )
        dirichlet_normalizer = sum(math.lgamma(float(a)) for a in alpha) - math.lgamma(alpha.sum())
        expected = -count_constant + dirichlet_normalizer
        expected -= float(np.dot(counts, np.log(theta)))
        expected -= float(np.dot(alpha - 1.0, np.log(theta)))
        expected += 0.5 * (len(alpha) - 1) * math.log(counts.sum())
        assert parameterized_multinomial_cost(counts, alpha) == pytest.approx(expected)

    def test_sparse_species_prior_favors_pure_counts(self):
        alpha = np.array([0.25, 0.25])
        pure = np.array([20.0, 0.0])
        mixed = np.array([10.0, 10.0])
        assert parameterized_multinomial_cost(pure, alpha) < parameterized_multinomial_cost(mixed, alpha)

    def test_center_species_prior_shrinks_pure_advantage(self):
        weak_alpha = np.array([0.25, 0.25])
        strong_alpha = np.array([100.0, 100.0])
        pure = np.array([20.0, 0.0])
        mixed = np.array([10.0, 10.0])
        assert parameterized_multinomial_cost(pure, weak_alpha) < parameterized_multinomial_cost(mixed, weak_alpha)
        assert parameterized_multinomial_cost(mixed, strong_alpha) < parameterized_multinomial_cost(pure, strong_alpha)

    def test_sparse_edge_prior_favors_narrow_distribution(self):
        alpha = uniform_alpha(10, kappa=1.0)
        narrow = np.zeros(10)
        narrow[3] = 100.0
        broad = np.full(10, 10.0)
        assert parameterized_multinomial_cost(narrow, alpha) < parameterized_multinomial_cost(broad, alpha)

    def test_center_edge_prior_shrinks_narrow_advantage(self):
        weak_alpha = uniform_alpha(10, kappa=1.0)
        strong_alpha = uniform_alpha(10, kappa=1000.0)
        narrow = np.zeros(10)
        narrow[3] = 100.0
        broad = np.full(10, 10.0)
        assert parameterized_multinomial_cost(narrow, weak_alpha) < parameterized_multinomial_cost(broad, weak_alpha)
        assert parameterized_multinomial_cost(broad, strong_alpha) < parameterized_multinomial_cost(narrow, strong_alpha)

    def test_posterior_mean_only_shrinks_sparse_advantage(self):
        weak_alpha = uniform_alpha(10, kappa=1.0)
        strong_alpha = uniform_alpha(10, kappa=1000.0)
        narrow = np.zeros(10)
        narrow[3] = 100.0
        broad = np.full(10, 10.0)
        weak_gap = parameterized_multinomial_cost(
            broad, weak_alpha, estimator="posterior_mean"
        ) - parameterized_multinomial_cost(
            narrow, weak_alpha, estimator="posterior_mean"
        )
        strong_gap = parameterized_multinomial_cost(
            broad, strong_alpha, estimator="posterior_mean"
        ) - parameterized_multinomial_cost(
            narrow, strong_alpha, estimator="posterior_mean"
        )
        assert strong_gap < weak_gap
