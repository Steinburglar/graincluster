"""Tests for greedy optimizer behavior."""

from __future__ import annotations

import numpy as np
import pytest

from graincluster.model.partition import partition_from_labels
from graincluster.optimizer.greedy import greedy_optimize, OptimizeResult
from tests.conftest import make_bin_scheme, make_two_domain_edges


def _two_domain_partition(n=5, alpha=0.5, gamma=0.5, lambda_cut=0.5):
    bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
    edges = make_two_domain_edges(
        n_a=n, n_b=n,
        pair_key="A-A", pair_type_idx=0,
        raw_a=2.0, raw_b=4.0, bridge_raw=3.0,
        bin_scheme=bs,
    )
    labels = np.arange(2 * n, dtype=int)  # each atom its own cluster
    return partition_from_labels(labels, edges, bs, alpha=alpha, gamma=gamma, lambda_cut=lambda_cut)


class TestGreedyOptimizer:
    def test_returns_optimize_result(self):
        p = _two_domain_partition()
        result = greedy_optimize(p)
        assert isinstance(result, OptimizeResult)

    def test_objective_does_not_increase(self):
        p = _two_domain_partition()
        result = greedy_optimize(p)
        assert result.objective_final <= result.objective_initial + 1e-9

    def test_converges_in_finite_passes(self):
        p = _two_domain_partition()
        result = greedy_optimize(p, max_passes=50)
        assert result.n_passes <= 50

    def test_no_moves_when_already_optimal(self):
        """A single-cluster partition (no cuts possible) should not move."""
        bs = make_bin_scheme(["A-A"], n_bins=4)
        from graincluster.graph.edge import EdgeRecord
        # Triangle graph, all same cluster.
        edges = [
            EdgeRecord(i=0, j=1, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
            EdgeRecord(i=1, j=2, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
            EdgeRecord(i=0, j=2, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
        ]
        labels = np.zeros(3, dtype=int)
        p = partition_from_labels(labels, edges, bs, gamma=10.0, lambda_cut=10.0)
        result = greedy_optimize(p, allow_splits=False)
        assert result.n_moves == 0

    def test_two_identical_domains_merge(self):
        """Two domains with same edge distribution and no cut penalty should merge."""
        bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
        edges = make_two_domain_edges(
            n_a=4, n_b=4,
            pair_key="A-A", pair_type_idx=0,
            raw_a=2.0, raw_b=2.0, bridge_raw=2.0,
            bin_scheme=bs,
        )
        labels = np.array([0] * 4 + [1] * 4, dtype=int)
        # alpha=0.0 gives zero self-information for dominant bins → no signal.
        # Use default alpha=0.5 so smoothing creates a positive reward for merging.
        p = partition_from_labels(labels, edges, bs, gamma=10.0, lambda_cut=0.0, alpha=0.5)
        greedy_optimize(p, allow_splits=False)
        assert p.n_clusters() == 1

    def test_two_different_domains_stay_separate(self):
        """Two domains with distinct edge distributions and high cut penalty stay separate."""
        bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
        edges = make_two_domain_edges(
            n_a=5, n_b=5,
            pair_key="A-A", pair_type_idx=0,
            raw_a=1.5, raw_b=4.5,  # very different distances
            bridge_raw=5.0,
            bin_scheme=bs,
        )
        labels = np.array([0] * 5 + [1] * 5, dtype=int)
        p = partition_from_labels(labels, edges, bs, gamma=0.0, lambda_cut=100.0)
        result = greedy_optimize(p, allow_splits=False)
        assert p.n_clusters() == 2

    def test_objective_monotone_decreasing(self):
        """Objective should decrease (or stay flat) across all passes."""
        p = _two_domain_partition(n=4)
        objs = [p.objective()]

        for _ in range(20):
            obj_before = p.objective()
            from graincluster.optimizer.greedy import greedy_optimize
            # Run one pass manually by checking score before applying.
            moved = False
            for atom in range(len(p.atom_labels)):
                src_id = int(p.atom_labels[atom])
                neighbor_clusters = set()
                for eidx in p._adj[atom]:
                    e = p.edges[eidx]
                    nbr = e.j if e.i == atom else e.i
                    neighbor_clusters.add(int(p.atom_labels[nbr]))
                neighbor_clusters.discard(src_id)
                for cid in neighbor_clusters:
                    delta = p.score_move(atom, cid)
                    if delta < -1e-10:
                        p.apply_move(atom, cid)
                        moved = True
                        break
            objs.append(p.objective())
            if not moved:
                break

        for i in range(1, len(objs)):
            assert objs[i] <= objs[i - 1] + 1e-9

    def test_max_passes_respected(self):
        p = _two_domain_partition(n=10)
        result = greedy_optimize(p, max_passes=2)
        assert result.n_passes <= 2

    def test_n_moves_tracked(self):
        p = _two_domain_partition()
        result = greedy_optimize(p)
        assert result.n_moves >= 0
