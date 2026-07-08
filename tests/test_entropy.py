"""Tests for entropy model: counts, smoothing, and objective terms."""

from __future__ import annotations

import math

import pytest

from graincluster.model.cluster import ClusterState
from graincluster.model.entropy import cluster_entropy, data_term, self_information


class TestClusterState:
    def test_add_edge_updates_count_and_N(self):
        c = ClusterState(cluster_id=0)
        c.add_edge(0, 2)
        assert c.counts[(0, 2)] == 1
        assert c.N == 1

    def test_add_edge_same_category_accumulates(self):
        c = ClusterState(cluster_id=0)
        c.add_edge(0, 2)
        c.add_edge(0, 2)
        assert c.counts[(0, 2)] == 2
        assert c.N == 2

    def test_remove_edge_decrements(self):
        c = ClusterState(cluster_id=0)
        c.add_edge(1, 3)
        c.add_edge(1, 3)
        c.remove_edge(1, 3)
        assert c.counts[(1, 3)] == 1
        assert c.N == 1

    def test_remove_edge_deletes_key_at_zero(self):
        c = ClusterState(cluster_id=0)
        c.add_edge(0, 0)
        c.remove_edge(0, 0)
        assert (0, 0) not in c.counts
        assert c.N == 0

    def test_remove_edge_raises_on_underflow(self):
        c = ClusterState(cluster_id=0)
        with pytest.raises(ValueError, match="Cannot remove"):
            c.remove_edge(0, 0)

    def test_is_empty(self):
        c = ClusterState(cluster_id=0)
        assert c.is_empty()
        c.atom_ids.add(5)
        assert not c.is_empty()

    def test_entropy_invalidated_on_add(self):
        c = ClusterState(cluster_id=0)
        c._entropy = 99.0
        c.add_edge(0, 0)
        assert c._entropy is None

    def test_entropy_invalidated_on_remove(self):
        c = ClusterState(cluster_id=0)
        c.add_edge(0, 0)
        c._entropy = 99.0
        c.remove_edge(0, 0)
        assert c._entropy is None


class TestClusterEntropy:
    """Verify entropy math against hand-computed values."""

    def test_empty_cluster_entropy_is_zero(self):
        c = ClusterState(cluster_id=0)
        assert cluster_entropy(c, M=10, alpha=0.5) == 0.0

    def test_uniform_distribution_maximum_entropy(self):
        # All M categories equally populated → H = log(M).
        M = 4
        c = ClusterState(cluster_id=0)
        for b in range(M):
            for _ in range(10):
                c.add_edge(0, b)
        h = cluster_entropy(c, M=M, alpha=0.0)
        assert h == pytest.approx(math.log(M), rel=1e-6)

    def test_single_category_low_entropy(self):
        # All edges in one category → near-zero entropy (smoothing raises it slightly).
        M = 10
        alpha = 0.5
        c = ClusterState(cluster_id=0)
        for _ in range(1000):
            c.add_edge(0, 0)
        h = cluster_entropy(c, M=M, alpha=alpha)
        # Should be much lower than log(M).
        assert h < 0.1 * math.log(M)

    def test_two_equal_categories_entropy(self):
        # 50/50 split over 2 categories, other M-2 empty → H ≈ log(2) with alpha=0.
        M = 2
        c = ClusterState(cluster_id=0)
        for _ in range(100):
            c.add_edge(0, 0)
            c.add_edge(0, 1)
        h = cluster_entropy(c, M=2, alpha=0.0)
        assert h == pytest.approx(math.log(2), rel=1e-4)

    def test_smoothing_raises_entropy_for_sparse_data(self):
        # With alpha > 0, absent categories get nonzero probability.
        M = 10
        c = ClusterState(cluster_id=0)
        c.add_edge(0, 0)
        h_no_smooth = cluster_entropy(c, M=M, alpha=0.0)
        h_smooth = cluster_entropy(c, M=M, alpha=0.5)
        assert h_smooth > h_no_smooth

    def test_entropy_nonnegative(self):
        M = 8
        c = ClusterState(cluster_id=0)
        for pt in range(2):
            for b in range(4):
                for _ in range(pt + b + 1):
                    c.add_edge(pt, b)
        assert cluster_entropy(c, M=M, alpha=0.5) >= 0.0

    def test_entropy_upper_bound_log_M(self):
        M = 20
        c = ClusterState(cluster_id=0)
        for b in range(M):
            c.add_edge(0, b)
        h = cluster_entropy(c, M=M, alpha=0.0)
        assert h <= math.log(M) + 1e-9

    def test_hand_computed_two_category(self):
        # 3 edges in (0,0), 1 edge in (0,1), M=4, alpha=0.
        # p0=3/4, p1=1/4, p2=p3=0
        # H = -(3/4*log(3/4) + 1/4*log(1/4))
        M = 4
        c = ClusterState(cluster_id=0)
        for _ in range(3):
            c.add_edge(0, 0)
        c.add_edge(0, 1)
        h = cluster_entropy(c, M=M, alpha=0.0)
        expected = -(3/4 * math.log(3/4) + 1/4 * math.log(1/4))
        assert h == pytest.approx(expected, rel=1e-9)

    def test_data_term_is_bayesian_marginal(self):
        """data_term == -log P(n | alpha, M) via lgamma formula."""
        from math import lgamma
        M = 6
        alpha = 0.5
        c = ClusterState(cluster_id=0)
        for b in range(3):
            for _ in range(5):
                c.add_edge(0, b)
        # n = {bin0: 5, bin1: 5, bin2: 5}, N = 15
        N = c.N
        expected = lgamma(N + alpha * M) - lgamma(alpha * M)
        expected -= sum(lgamma(cnt + alpha) - lgamma(alpha) for cnt in c.counts.values())
        dt = data_term(c, M=M, alpha=alpha)
        assert dt == pytest.approx(expected, rel=1e-9)

    def test_data_term_zero_N(self):
        """Empty cluster has code length 0."""
        M = 6
        c = ClusterState(cluster_id=0)
        assert data_term(c, M=M, alpha=0.5) == 0.0

    def test_data_term_zero_count_categories_ignored(self):
        """Zero-count categories contribute 0 to L_Bayes (lgamma cancels)."""
        from math import lgamma
        M = 10
        alpha = 0.5
        # only 2 of 10 bins occupied
        c = ClusterState(cluster_id=0)
        c.add_edge(0, 0)
        c.add_edge(0, 0)
        c.add_edge(0, 1)
        N = c.N
        expected = lgamma(N + alpha * M) - lgamma(alpha * M)
        expected -= sum(lgamma(cnt + alpha) - lgamma(alpha) for cnt in c.counts.values())
        assert data_term(c, M=M, alpha=alpha) == pytest.approx(expected, rel=1e-9)


class TestSelfInformation:
    def test_self_information_equals_neg_log_prob(self):
        M = 4
        alpha = 0.5
        c = ClusterState(cluster_id=0)
        c.add_edge(0, 0)
        c.add_edge(0, 0)
        c.add_edge(0, 1)
        # p̃(0,0) = (2 + 0.5) / (3 + 0.5*4) = 2.5 / 5 = 0.5
        # I = -log(0.5) = log(2)
        I = self_information(0, 0, c, M=M, alpha=alpha)
        assert I == pytest.approx(math.log(2), rel=1e-9)

    def test_self_information_absent_category(self):
        M = 4
        alpha = 0.5
        c = ClusterState(cluster_id=0)
        c.add_edge(0, 0)
        # p̃(0,1) = (0 + 0.5) / (1 + 0.5*4) = 0.5 / 3
        # I = -log(0.5/3) = log(6)
        I = self_information(0, 1, c, M=M, alpha=alpha)
        assert I == pytest.approx(math.log(6), rel=1e-9)

    def test_self_information_positive(self):
        M = 10
        c = ClusterState(cluster_id=0)
        for b in range(5):
            c.add_edge(0, b)
        for pt in range(2):
            for b in range(5):
                assert self_information(pt, b, c, M=M, alpha=0.5) > 0.0

    def test_self_information_does_not_mutate_cluster(self):
        M = 4
        c = ClusterState(cluster_id=0)
        c.add_edge(0, 0)
        N_before = c.N
        counts_before = dict(c.counts)
        self_information(0, 0, c, M=M, alpha=0.5)
        assert c.N == N_before
        assert c.counts == counts_before

    def test_one_species_reduces_correctly(self):
        # One species: only pair type 0, M=n_bins.
        # Should give same entropy as a standard single-type histogram.
        M = 5  # 1 pair type, 5 bins
        alpha = 0.5
        c = ClusterState(cluster_id=0)
        counts_raw = [10, 5, 2, 0, 3]
        for b, cnt in enumerate(counts_raw):
            for _ in range(cnt):
                c.add_edge(0, b)
        N = sum(counts_raw)
        denom = N + alpha * M
        expected_h = -sum(
            ((cnt + alpha) / denom) * math.log((cnt + alpha) / denom)
            for cnt in counts_raw
        )
        h = cluster_entropy(c, M=M, alpha=alpha)
        assert h == pytest.approx(expected_h, rel=1e-9)
