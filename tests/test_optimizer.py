"""Tests for greedy optimizer behavior."""

from __future__ import annotations

import numpy as np
import pytest

from graincluster.model.partition import partition_from_labels
from graincluster.optimizer.greedy import greedy_optimize, OptimizeResult
from tests.conftest import make_bin_scheme, make_two_domain_edges


def _two_domain_partition(n=5, alpha=0.5, gamma=0.5, beta=0.5):
    bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
    edges = make_two_domain_edges(
        n_a=n, n_b=n,
        pair_key="A-A", pair_type_idx=0,
        raw_a=2.0, raw_b=4.0, bridge_raw=3.0,
        bin_scheme=bs,
    )
    labels = np.arange(2 * n, dtype=int)  # each atom its own cluster
    return partition_from_labels(labels, edges, bs, alpha=alpha, gamma=gamma, beta=beta)


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
        p = partition_from_labels(labels, edges, bs, gamma=10.0, beta=0.9)
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
        p = partition_from_labels(labels, edges, bs, gamma=10.0, beta=0.0, alpha=0.5)
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
        p = partition_from_labels(labels, edges, bs, gamma=0.0, beta=0.99)
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


class TestConnectivitySplit:
    def _line_partition(self):
        """4-atom line graph (0-1-2-3), all in cluster 0."""
        from graincluster.graph.edge import EdgeRecord
        bs = make_bin_scheme(["A-A"], n_bins=4)
        edges = [
            EdgeRecord(i=0, j=1, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
            EdgeRecord(i=1, j=2, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
            EdgeRecord(i=2, j=3, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
        ]
        labels = np.zeros(4, dtype=int)
        return partition_from_labels(labels, edges, bs, gamma=1.0, beta=0.5)

    def test_connected_cluster_not_split(self):
        from graincluster.graph.edge import EdgeRecord
        from graincluster.optimizer.greedy import _split_if_disconnected
        bs = make_bin_scheme(["A-A"], n_bins=4)
        edges = [
            EdgeRecord(i=0, j=1, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
            EdgeRecord(i=1, j=2, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
            EdgeRecord(i=0, j=2, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
        ]
        labels = np.zeros(3, dtype=int)
        p = partition_from_labels(labels, edges, bs)
        n_new = _split_if_disconnected(p, 0)
        assert n_new == 0
        assert p.n_clusters() == 1

    def test_disconnected_cluster_splits(self):
        """Moving bridging atom disconnects the source cluster → split into 2."""
        from graincluster.optimizer.greedy import _split_if_disconnected
        p = self._line_partition()
        # Remove bridge atom 1; cluster 0 becomes {0, 2, 3} — disconnected.
        p.apply_move(1, p.new_cluster_id())
        n_new = _split_if_disconnected(p, 0)
        assert n_new == 1
        # 3 clusters: {1}, {0}, {2,3}
        assert p.n_clusters() == 3
        # Atoms 2 and 3 stay together; atom 0 is isolated.
        assert int(p.atom_labels[2]) == int(p.atom_labels[3])
        assert int(p.atom_labels[0]) != int(p.atom_labels[2])

    def test_edge_counts_correct_after_split(self):
        """After split: isolated {0} has N=0; component {2,3} has N=1."""
        from graincluster.optimizer.greedy import _split_if_disconnected
        p = self._line_partition()
        p.apply_move(1, p.new_cluster_id())
        _split_if_disconnected(p, 0)
        cid_0 = int(p.atom_labels[0])
        cid_2 = int(p.atom_labels[2])
        assert p.clusters[cid_0].N == 0
        assert p.clusters[cid_2].N == 1

    def test_objective_consistent_after_split(self):
        """partition.objective() is self-consistent after a connectivity split."""
        from graincluster.optimizer.greedy import _split_if_disconnected
        from graincluster.model.entropy import data_term
        import pytest as _pytest
        p = self._line_partition()
        p.apply_move(1, p.new_cluster_id())
        _split_if_disconnected(p, 0)
        M = p._M
        L_data = (1.0 - p.beta) * sum(data_term(c, M, p.alpha) for c in p.clusters.values())
        L_cut = p.beta * sum(
            e.cut_cost for e in p.edges if p.atom_labels[e.i] != p.atom_labels[e.j]
        )
        expected = L_data + p.gamma * p.n_clusters() + L_cut
        assert p.objective() == _pytest.approx(expected, abs=1e-9)

    def test_optimizer_clusters_are_connected(self):
        """All clusters produced by greedy_optimize are connected subgraphs."""
        p = _two_domain_partition()
        greedy_optimize(p)
        for cid, c in p.clusters.items():
            if len(c.atom_ids) <= 1:
                continue
            reachable: set[int] = set()
            stack = [next(iter(c.atom_ids))]
            while stack:
                atom = stack.pop()
                if atom in reachable:
                    continue
                reachable.add(atom)
                for eidx in p._adj[atom]:
                    e = p.edges[eidx]
                    nbr = e.j if e.i == atom else e.i
                    if int(p.atom_labels[nbr]) == cid and nbr not in reachable:
                        stack.append(nbr)
            assert reachable == c.atom_ids, f"Cluster {cid} is disconnected"
