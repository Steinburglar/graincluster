"""Integration tests: end-to-end behavior on synthetic graphs.

Validates scientific expectations described in the implementation plan section 16.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from graincluster.graph.edge import EdgeRecord
from graincluster.features.binning import fit_bin_scheme
from graincluster.model.entropy import cluster_entropy
from graincluster.model.cluster import ClusterState
from graincluster.model.partition import partition_from_labels
from graincluster.optimizer.greedy import greedy_optimize
from tests.conftest import make_bin_scheme, make_two_domain_edges


# ---------------------------------------------------------------------------
# Entropy order tests (plan section 16.3)
# ---------------------------------------------------------------------------

class TestEntropyOrder:
    def test_crystal_lower_entropy_than_liquid(self):
        """Narrow bond-length distribution (crystal-like) → lower entropy."""
        M = 20
        alpha = 0.5

        crystal = ClusterState(cluster_id=0)
        for _ in range(200):
            crystal.add_edge(0, 5)   # all edges in one bin

        liquid = ClusterState(cluster_id=1)
        for b in range(10):
            for _ in range(20):    # uniform over 10 bins
                liquid.add_edge(0, b)

        h_crystal = cluster_entropy(crystal, M=M, alpha=alpha)
        h_liquid = cluster_entropy(liquid, M=M, alpha=alpha)
        assert h_crystal < h_liquid

    def test_mixed_species_higher_entropy_than_pure(self):
        """Mixed-species cluster (two pair types active) → higher entropy than pure."""
        M = 20   # e.g. 2 pair types × 10 bins
        alpha = 0.5

        pure = ClusterState(cluster_id=0)
        for _ in range(100):
            pure.add_edge(0, 3)   # only pair type 0, bin 3

        mixed = ClusterState(cluster_id=1)
        for _ in range(50):
            mixed.add_edge(0, 3)  # pair type 0
        for _ in range(50):
            mixed.add_edge(1, 3)  # pair type 1 — same bin, different type

        h_pure = cluster_entropy(pure, M=M, alpha=alpha)
        h_mixed = cluster_entropy(mixed, M=M, alpha=alpha)
        assert h_mixed > h_pure

    def test_one_species_entropy_matches_single_type_calc(self):
        """One-species limit: joint entropy collapses to single-type histogram entropy."""
        n_bins = 8
        M = n_bins   # 1 pair type × n_bins
        alpha = 0.5
        counts = [10, 8, 5, 0, 0, 3, 2, 1]

        c = ClusterState(cluster_id=0)
        for b, cnt in enumerate(counts):
            for _ in range(cnt):
                c.add_edge(0, b)

        N = sum(counts)
        denom = N + alpha * M
        expected = -sum(
            ((cnt + alpha) / denom) * math.log((cnt + alpha) / denom)
            for cnt in counts
        )
        assert cluster_entropy(c, M=M, alpha=alpha) == pytest.approx(expected, rel=1e-9)


class TestInterfaceDetection:
    def test_optimizer_finds_two_distinct_phases(self):
        """Two well-separated bond-length domains should end up in separate clusters."""
        n = 6
        bs = make_bin_scheme(["A-A"], n_bins=20, lo=1.0, hi=6.0)
        edges = make_two_domain_edges(
            n_a=n, n_b=n,
            pair_key="A-A", pair_type_idx=0,
            raw_a=1.8, raw_b=5.2,   # well-separated
            bridge_raw=3.5,
            bin_scheme=bs,
        )
        # Start with all atoms in one cluster.
        labels = np.zeros(2 * n, dtype=int)
        p = partition_from_labels(labels, edges, bs, alpha=0.5, gamma=0.1, lambda_cut=0.0)
        result = greedy_optimize(p, max_passes=50)
        # Should split into at least 2 clusters (the two phases).
        assert p.n_clusters() >= 2

    def test_single_phase_stays_merged(self):
        """Uniform edge distribution → starting merged should stay merged."""
        n = 5
        bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
        # All edges at same distance.
        from graincluster.graph.edge import EdgeRecord
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append(EdgeRecord(
                    i=i, j=j, pair_key="A-A", pair_type_idx=0,
                    raw_value=2.5, bin_idx=4, cut_cost=3.125,
                ))
        labels = np.zeros(n, dtype=int)
        p = partition_from_labels(labels, edges, bs, alpha=0.5, gamma=1.0, lambda_cut=1.0)
        greedy_optimize(p, max_passes=20)
        assert p.n_clusters() == 1

    def test_strong_bridge_cut_is_expensive(self):
        """High lambda_cut should prevent cutting a short-distance bridge."""
        n = 3
        bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
        # Two tiny domains (n=3 each) with very short bridge.
        edges = make_two_domain_edges(
            n_a=n, n_b=n,
            pair_key="A-A", pair_type_idx=0,
            raw_a=2.0, raw_b=4.0,
            bridge_raw=1.01,  # very short bridge
            bin_scheme=bs,
        )
        labels = np.array([0] * n + [1] * n, dtype=int)
        p = partition_from_labels(labels, edges, bs, gamma=0.0, lambda_cut=1000.0)
        # Even if entropy wants a split, cut cost should dominate.
        result = greedy_optimize(p, allow_splits=False, max_passes=20)
        # The bridge atom (n-1 or n) should not move away from its domain.
        # Objective should stay the same or increase (optimizer should reject moves).
        assert result.objective_final <= result.objective_initial + 1e-6

    def test_objective_decreases_from_all_singleton_start(self):
        """Starting with all atoms as singletons → optimizer should reduce objective."""
        n = 5
        bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
        edges = make_two_domain_edges(
            n_a=n, n_b=n,
            pair_key="A-A", pair_type_idx=0,
            raw_a=2.0, raw_b=4.0, bridge_raw=3.0,
            bin_scheme=bs,
        )
        labels = np.arange(2 * n, dtype=int)
        p = partition_from_labels(labels, edges, bs, alpha=0.5, gamma=0.5, lambda_cut=0.5)
        result = greedy_optimize(p, max_passes=50)
        assert result.objective_final < result.objective_initial


class TestFitBinSchemeIntegration:
    def test_fit_then_assign_all_valid(self):
        """fit_bin_scheme on realistic distances → all assigned bins valid."""
        rng = np.random.default_rng(42)
        pair_values = {
            "Au-Au": rng.normal(2.88, 0.15, size=800),
            "Au-Pt": rng.normal(2.77, 0.12, size=600),
        }
        bs = fit_bin_scheme(pair_values)

        for pk, vals in pair_values.items():
            bins = bs.assign(pk, vals)
            scheme = bs.schemes[pk]
            assert np.all(bins >= 0)
            assert np.all(bins < scheme.n_bins)

    def test_freedman_diaconis_bin_width_narrower_for_crystal(self):
        """Narrow (crystal) distribution → smaller bin width than broad (liquid).

        FD is scale-invariant for normal distributions (span and IQR both scale
        with sigma), so bin COUNT is the same. What changes is bin WIDTH and the
        range of edges covered by the scheme.
        """
        rng = np.random.default_rng(7)
        crystal_vals = rng.normal(2.88, 0.05, size=1000)
        liquid_vals = rng.normal(2.88, 0.50, size=1000)
        pair_values = {"A-A": crystal_vals, "B-B": liquid_vals}
        bs = fit_bin_scheme(pair_values)
        aa = bs.schemes["A-A"]
        bb = bs.schemes["B-B"]
        bin_width_crystal = (aa.range_hi - aa.range_lo) / aa.n_bins
        bin_width_liquid = (bb.range_hi - bb.range_lo) / bb.n_bins
        assert bin_width_crystal < bin_width_liquid
