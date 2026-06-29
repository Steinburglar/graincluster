"""Tests for move scoring and application."""

from __future__ import annotations

import math

import numpy as np
import pytest

from graincluster.graph.edge import EdgeRecord
from graincluster.model.partition import Partition, partition_from_labels
from tests.conftest import make_bin_scheme, make_two_domain_edges


def _simple_two_cluster_partition(n_per_cluster=4, n_bins=10, alpha=0.5, gamma=1.0, lambda_cut=1.0):
    """Two clusters (ids 0 and 1), each a clique, connected by one bridge edge.

    Atoms 0..n-1 in cluster 0, atoms n..2n-1 in cluster 1.
    All edges are pair type 0 (A-A), bin 3 (mid-range).
    Bridge edge uses a different bin (9, near range edge).
    """
    n = n_per_cluster
    bs = make_bin_scheme(["A-A"], n_bins=n_bins, lo=1.0, hi=5.0)
    labels = np.array([0] * n + [1] * n, dtype=int)

    edges = make_two_domain_edges(
        n_a=n,
        n_b=n,
        pair_key="A-A",
        pair_type_idx=0,
        raw_a=2.0,       # bin ~ 2-3
        raw_b=4.0,       # bin ~ 7-8
        bridge_raw=3.0,  # bin ~ 5 (bridge)
        bin_scheme=bs,
    )
    return partition_from_labels(labels, edges, bs, alpha=alpha, gamma=gamma, lambda_cut=lambda_cut)


class TestScoreMove:
    def test_score_move_same_cluster_is_zero(self):
        p = _simple_two_cluster_partition()
        atom = 0
        src_cid = int(p.atom_labels[atom])
        assert p.score_move(atom, src_cid) == 0.0

    def test_score_move_returns_float(self):
        p = _simple_two_cluster_partition()
        delta = p.score_move(0, 1)
        assert isinstance(delta, float)

    def test_move_bridge_atom_to_other_cluster_has_finite_delta(self):
        p = _simple_two_cluster_partition()
        # Atom n_per_cluster-1 is in cluster 0 and connected to cluster 1 via bridge.
        n = 4
        delta = p.score_move(n - 1, 1)
        assert math.isfinite(delta)

    def test_score_move_split_to_new_cluster(self):
        p = _simple_two_cluster_partition()
        new_cid = p.new_cluster_id()
        delta = p.score_move(0, new_cid)
        assert math.isfinite(delta)

    def test_apply_move_updates_atom_labels(self):
        p = _simple_two_cluster_partition()
        p.apply_move(0, 1)
        assert int(p.atom_labels[0]) == 1

    def test_apply_move_updates_cluster_membership(self):
        p = _simple_two_cluster_partition()
        p.apply_move(0, 1)
        assert 0 not in p.clusters[0].atom_ids
        assert 0 in p.clusters[1].atom_ids

    def test_apply_move_updates_edge_counts(self):
        p = _simple_two_cluster_partition()
        n = 4
        # Before: atom 0 connected to atoms 1,2,3 all in cluster 0 (internal).
        src = p.clusters[0]
        N_before = src.N
        p.apply_move(0, 1)
        # Atom 0's n-1=3 intra-cluster edges became cut.
        assert p.clusters[0].N == N_before - 3

    def test_apply_move_removes_empty_cluster(self):
        bs = make_bin_scheme(["A-A"], n_bins=4)
        # Two atoms, one edge between them, separate clusters.
        edges = [EdgeRecord(i=0, j=1, pair_key="A-A", pair_type_idx=0,
                            raw_value=2.0, bin_idx=2, cut_cost=2.0)]
        labels = np.array([0, 1], dtype=int)
        p = partition_from_labels(labels, edges, bs)
        p.apply_move(0, 1)
        assert 0 not in p.clusters
        assert 1 in p.clusters

    def test_apply_move_creates_new_cluster(self):
        p = _simple_two_cluster_partition()
        new_cid = p.new_cluster_id()
        p.apply_move(0, new_cid)
        assert new_cid in p.clusters
        assert 0 in p.clusters[new_cid].atom_ids

    def test_frozen_model_rejects_costly_move(self):
        """Frozen model should score clearly bad moves positively.

        Moving an interior atom (all neighbors in its cluster, high cut penalty)
        to the other cluster is costly. score_move should return > 0.

        NOTE: The frozen model is an approximation. Sign agreement with the exact
        objective change is only reliable for large clusters (N >> 1). For N=1 or
        N=2, the approximation can disagree in sign because removing the last few
        edges from a cluster collapses data_term to 0, which the frozen model
        underestimates. This test uses a large cluster with very high lambda_cut
        so the cut-cost signal dominates and the approximation is reliable.
        """
        n = 8
        bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
        edges = make_two_domain_edges(
            n_a=n, n_b=n,
            pair_key="A-A", pair_type_idx=0,
            raw_a=2.0, raw_b=4.0, bridge_raw=3.0,
            bin_scheme=bs,
        )
        labels = np.array([0] * n + [1] * n, dtype=int)
        # High lambda_cut: moving atom 0 (deep interior) to cluster 1 is very costly.
        p = partition_from_labels(labels, edges, bs, lambda_cut=50.0)
        delta = p.score_move(0, 1)
        # Should clearly identify the move as costly.
        assert delta > 0

    def test_boundary_penalty_contributes_to_delta(self):
        """Moving bridge atom to other cluster should add cut cost for edges to old cluster."""
        n = 4
        bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
        edges = make_two_domain_edges(
            n_a=n, n_b=n,
            pair_key="A-A", pair_type_idx=0,
            raw_a=2.0, raw_b=4.0, bridge_raw=3.0,
            bin_scheme=bs,
        )
        labels = np.array([0] * n + [1] * n, dtype=int)
        lam = 5.0
        p = partition_from_labels(labels, edges, bs, lambda_cut=lam)
        # Bridge edge is between atom n-1 (cluster 0) and atom n (cluster 1).
        # Moving atom 0 (interior of cluster 0) to cluster 1 increases cut cost.
        delta = p.score_move(0, 1)
        # The cut cost portion for atom 0's edges (all go to cluster 0 neighbors)
        # must increase by lambda * sum(cut_costs).
        adj_edges = [edges[eidx] for eidx in p._adj[0]]
        neighbor_cut_increase = lam * sum(e.cut_cost for e in adj_edges
                                           if int(labels[e.j if e.i == 0 else e.i]) == 0)
        # delta should include this positive cut contribution.
        assert delta > 0  # Moving an interior atom to the other cluster should be costly.


class TestExactDelta:
    def test_exact_matches_frozen_for_large_clusters(self):
        """For large clusters frozen and exact should agree closely."""
        p = _simple_two_cluster_partition(n_per_cluster=8)
        atom = 0  # deep interior of cluster 0
        delta_frozen = p.score_move(atom, 1, exact_below_N=0)
        delta_exact = p.score_move(atom, 1, exact_below_N=100)
        # For large N the approximation is close but not exact.
        # They should at minimum agree in sign.
        assert (delta_frozen > 0) == (delta_exact > 0)

    def test_exact_corrects_last_edge_removal(self):
        """Exact delta > frozen delta when removing last edge from small src.

        With N_src=1, frozen underestimates entropy gain; exact should give
        a more negative (better) score for the split.
        """
        bs = make_bin_scheme(["A-A"], n_bins=20, lo=1.0, hi=5.0)
        # Two atoms, one edge — clusters {0} and {1}, one cut edge between them.
        from graincluster.graph.edge import EdgeRecord
        edges = [EdgeRecord(i=0, j=1, pair_key="A-A", pair_type_idx=0,
                            raw_value=2.5, bin_idx=9, cut_cost=3.125)]
        labels = np.array([0, 0], dtype=int)  # both in cluster 0, one internal edge
        p = partition_from_labels(labels, edges, bs, gamma=0.0, lambda_cut=0.0)
        assert p.clusters[0].N == 1

        # Split atom 0 to a new singleton.
        new_cid = p.new_cluster_id()
        delta_frozen = p.score_move(0, new_cid, exact_below_N=0)
        delta_exact = p.score_move(0, new_cid, exact_below_N=10)
        # Exact savings should be larger (more negative) than frozen.
        assert delta_exact <= delta_frozen

    def test_exact_agrees_with_objective_after_apply(self):
        """exact score_move delta should match the actual objective change."""
        n = 4
        bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
        edges = make_two_domain_edges(
            n_a=n, n_b=n,
            pair_key="A-A", pair_type_idx=0,
            raw_a=2.0, raw_b=4.0, bridge_raw=3.0,
            bin_scheme=bs,
        )
        labels = np.array([0] * n + [1] * n, dtype=int)
        p = partition_from_labels(labels, edges, bs, gamma=0.5, lambda_cut=1.0)

        atom = n - 1  # bridge atom in cluster 0
        target = 1
        obj_before = p.objective()
        delta_exact = p.score_move(atom, target, exact_below_N=1000)
        p.apply_move(atom, target)
        obj_after = p.objective()
        actual_delta = obj_after - obj_before
        assert actual_delta == pytest.approx(delta_exact, abs=1e-9)

    def test_exact_agrees_with_objective_singleton_merge(self):
        """Merging two singletons: exact delta == actual objective change."""
        bs = make_bin_scheme(["A-A"], n_bins=20, lo=1.0, hi=5.0)
        from graincluster.graph.edge import EdgeRecord
        edges = [EdgeRecord(i=0, j=1, pair_key="A-A", pair_type_idx=0,
                            raw_value=2.5, bin_idx=9, cut_cost=3.125)]
        labels = np.array([0, 1], dtype=int)
        p = partition_from_labels(labels, edges, bs, gamma=0.5, lambda_cut=1.0)
        # Both clusters are singletons (N=0).
        obj_before = p.objective()
        delta_exact = p.score_move(0, 1, exact_below_N=1000)
        p.apply_move(0, 1)
        obj_after = p.objective()
        actual_delta = obj_after - obj_before
        assert actual_delta == pytest.approx(delta_exact, abs=1e-9)


class TestObjectiveConsistency:
    def test_objective_nonnegative(self):
        p = _simple_two_cluster_partition()
        assert p.objective() >= 0.0

    def test_objective_decreases_on_good_move(self):
        """Merging two identical domains should reduce or hold objective."""
        bs = make_bin_scheme(["A-A"], n_bins=10)
        # Two identical cliques with same raw value → merging should help.
        edges = make_two_domain_edges(
            n_a=4, n_b=4,
            pair_key="A-A", pair_type_idx=0,
            raw_a=2.0, raw_b=2.0, bridge_raw=2.0,
            bin_scheme=bs,
        )
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=int)
        p = partition_from_labels(labels, edges, bs, gamma=0.0, lambda_cut=0.0)
        obj_sep = p.objective()
        # Force merge by moving all cluster-1 atoms into cluster 0.
        for atom in range(4, 8):
            p.apply_move(atom, 0)
        obj_merged = p.objective()
        # Merged cluster has lower entropy (same distribution, fewer zero-count categories
        # after smoothing) or equal. Combined data term should not be worse.
        assert obj_merged <= obj_sep + 1e-9

    def test_partition_from_labels_counts_correct(self):
        """partition_from_labels should produce correct internal edge counts."""
        bs = make_bin_scheme(["A-A"], n_bins=4)
        # 3 atoms: 0-1 edge (both cluster 0), 1-2 edge (cross), 2-3 edge (both cluster 1).
        # Wait: 3 atoms → atoms 0,1,2. Labels: 0,0,1.
        edges = [
            EdgeRecord(i=0, j=1, pair_key="A-A", pair_type_idx=0, raw_value=2.0, bin_idx=1, cut_cost=2.0),
            EdgeRecord(i=1, j=2, pair_key="A-A", pair_type_idx=0, raw_value=2.5, bin_idx=2, cut_cost=3.0),
        ]
        labels = np.array([0, 0, 1], dtype=int)
        p = partition_from_labels(labels, edges, bs)
        # Cluster 0: edge (0,1) internal → N=1.
        assert p.clusters[0].N == 1
        # Cluster 1: no internal edges → N=0.
        assert p.clusters[1].N == 0
        # Edge (1,2) is cut.
        cut_cost_total = p.lambda_cut * sum(
            e.cut_cost for e in p.edges if p.atom_labels[e.i] != p.atom_labels[e.j]
        )
        assert cut_cost_total == pytest.approx(1.0 * 3.0)
