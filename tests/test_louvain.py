"""Tests for Louvain-style cluster-merge phase and outer loop."""

from __future__ import annotations

import numpy as np
import pytest

from graincluster.graph.edge import EdgeRecord
from graincluster.model.entropy import data_term, cluster_entropy
from graincluster.model.partition import partition_from_labels
from graincluster.optimizer.louvain import cluster_merge_sweep, louvain_optimize
from tests.conftest import make_bin_scheme, make_two_domain_edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _two_identical_clusters(n=6, alpha=0.5, gamma=0.5, beta=0.5):
    """Two clusters with identical edge distributions — cannot merge by atom moves."""
    bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
    edges = make_two_domain_edges(
        n_a=n, n_b=n,
        pair_key="A-A", pair_type_idx=0,
        raw_a=2.0, raw_b=2.0, bridge_raw=2.0,
        bin_scheme=bs,
    )
    labels = np.array([0] * n + [1] * n, dtype=int)
    return partition_from_labels(
        labels, edges, bs, alpha=alpha, gamma=gamma, beta=beta
    )


def _two_distinct_clusters(n=5, alpha=0.01, gamma=0.5, beta=0.01):
    """Two clusters with very different edge distributions.

    alpha=0.01 makes Dirichlet smoothing weak so entropy difference is large.
    beta=0.01 keeps cut-savings small so entropy dominates ΔL.
    Under these parameters score_cluster_merge > 0 (don't merge).
    """
    bs = make_bin_scheme(["A-A"], n_bins=20, lo=1.0, hi=6.0)
    edges = make_two_domain_edges(
        n_a=n, n_b=n,
        pair_key="A-A", pair_type_idx=0,
        raw_a=1.5, raw_b=5.5,
        bridge_raw=3.5,
        bin_scheme=bs,
    )
    labels = np.array([0] * n + [1] * n, dtype=int)
    return partition_from_labels(
        labels, edges, bs, alpha=alpha, gamma=gamma, beta=beta
    )


def _identical_clusters_cut_blocked(n=6):
    """Two identical clusters where atom moves are blocked by cut cost.

    beta=0.9: cut penalty dominates for single-atom moves (net 4 new cuts × 2.0
    nats each >> entropy savings), so greedy makes zero moves. But cluster merge
    absorbs the single bridge cut and removes a cluster → ΔL < 0.
    """
    bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
    edges = make_two_domain_edges(
        n_a=n, n_b=n,
        pair_key="A-A", pair_type_idx=0,
        raw_a=2.0, raw_b=2.0, bridge_raw=2.0,
        bin_scheme=bs,
    )
    labels = np.array([0] * n + [1] * n, dtype=int)
    return partition_from_labels(
        labels, edges, bs, alpha=0.5, gamma=100.0, beta=0.9
    )


# ---------------------------------------------------------------------------
# score_cluster_merge
# ---------------------------------------------------------------------------

class TestScoreClusterMerge:
    def test_returns_float(self):
        p = _two_identical_clusters()
        delta = p.score_cluster_merge(0, 1)
        assert isinstance(delta, float)

    def test_identical_clusters_merge_is_favorable(self):
        """Same distribution plus cluster penalty → merge should have ΔL < 0."""
        p = _two_identical_clusters(beta=0.0, gamma=150.0)
        delta = p.score_cluster_merge(0, 1)
        assert delta < 0.0

    def test_distinct_clusters_entropy_dominated_stay_separate(self):
        """Large entropy increase + weak cut savings → ΔL > 0 (don't merge).

        alpha=0.01 weakens smoothing so entropy difference is ~16 nats.
        beta=0.01 makes cut savings tiny (~0.06 nats for 1 bridge edge).
        ΔL_data >> ΔL_cut + ΔL_K so the merge is unfavorable.
        """
        p = _two_distinct_clusters()
        delta = p.score_cluster_merge(0, 1)
        assert delta > 0.0

    def test_score_matches_actual_objective_change(self):
        """score_cluster_merge ΔL must equal exact objective change after apply."""
        p = _two_identical_clusters()
        obj_before = p.objective()
        delta_scored = p.score_cluster_merge(0, 1)
        p.apply_cluster_merge(src_cid=0, tgt_cid=1)
        obj_after = p.objective()
        assert obj_after - obj_before == pytest.approx(delta_scored, abs=1e-9)

    def test_score_matches_actual_objective_change_distinct(self):
        p = _two_distinct_clusters(beta=1.0, gamma=0.5)
        obj_before = p.objective()
        delta_scored = p.score_cluster_merge(0, 1)
        p.apply_cluster_merge(src_cid=0, tgt_cid=1)
        obj_after = p.objective()
        assert obj_after - obj_before == pytest.approx(delta_scored, abs=1e-9)

    def test_score_matches_actual_objective_change_cluster_count_prior(self):
        p = _two_identical_clusters(beta=1.0, gamma=0.0)
        p.structure_prior_mode = "cluster_count"
        p.cluster_count_prior_mean = 1.0
        p.cluster_count_prior_tau = -1.0
        obj_before = p.objective()
        delta_scored = p.score_cluster_merge(0, 1)
        p.apply_cluster_merge(src_cid=0, tgt_cid=1)
        obj_after = p.objective()
        assert obj_after - obj_before == pytest.approx(delta_scored, abs=1e-9)


# ---------------------------------------------------------------------------
# apply_cluster_merge
# ---------------------------------------------------------------------------

class TestApplyClusterMerge:
    def test_src_removed(self):
        p = _two_identical_clusters()
        p.apply_cluster_merge(src_cid=0, tgt_cid=1)
        assert 0 not in p.clusters
        assert 1 in p.clusters

    def test_atom_labels_updated(self):
        n = 4
        p = _two_identical_clusters(n=n)
        p.apply_cluster_merge(src_cid=0, tgt_cid=1)
        # All atoms (0..2n-1) should now be in cluster 1.
        assert all(int(p.atom_labels[i]) == 1 for i in range(2 * n))

    def test_atom_ids_merged(self):
        n = 5
        p = _two_identical_clusters(n=n)
        p.apply_cluster_merge(src_cid=0, tgt_cid=1)
        assert p.clusters[1].atom_ids == set(range(2 * n))

    def test_edge_counts_correct_after_merge(self):
        """After merge, tgt.N = N_a + N_b + n_cross_edges."""
        n = 4
        p = _two_identical_clusters(n=n)
        N_a = p.clusters[0].N
        N_b = p.clusters[1].N
        # Count cross edges manually.
        n_cross = sum(
            1 for e in p.edges
            if (int(p.atom_labels[e.i]) == 0 and int(p.atom_labels[e.j]) == 1)
            or (int(p.atom_labels[e.i]) == 1 and int(p.atom_labels[e.j]) == 0)
        )
        p.apply_cluster_merge(src_cid=0, tgt_cid=1)
        assert p.clusters[1].N == N_a + N_b + n_cross

    def test_objective_consistent_after_merge(self):
        """Partition objective after apply_cluster_merge is self-consistent."""
        p = _two_identical_clusters()
        p.apply_cluster_merge(src_cid=0, tgt_cid=1)
        # Recompute parameterized objective from scratch and compare.
        L_data = (1.0 - p.beta) * sum(p._cluster_data_term(c) for c in p.clusters.values())
        K = p.n_clusters()
        L_cut = p.beta * sum(
            e.cut_cost
            for e in p.edges
            if p.atom_labels[e.i] != p.atom_labels[e.j]
        )
        expected = L_data + p.gamma * K + L_cut
        assert p.objective() == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# cluster_merge_sweep
# ---------------------------------------------------------------------------

class TestClusterMergeSweep:
    def test_returns_int(self):
        p = _two_identical_clusters()
        n = cluster_merge_sweep(p)
        assert isinstance(n, int)

    def test_identical_clusters_merged_in_one_sweep(self):
        """Two identical clusters with enough cluster penalty → merged in one sweep."""
        p = _two_identical_clusters(beta=0.0, gamma=150.0)
        n = cluster_merge_sweep(p)
        assert n == 1
        assert p.n_clusters() == 1

    def test_distinct_clusters_not_merged(self):
        """Entropy increase dominates cut savings → not merged."""
        p = _two_distinct_clusters()
        n = cluster_merge_sweep(p)
        assert n == 0
        assert p.n_clusters() == 2

    def test_objective_does_not_increase(self):
        p = _two_identical_clusters()
        obj_before = p.objective()
        cluster_merge_sweep(p)
        assert p.objective() <= obj_before + 1e-9

    def test_single_cluster_no_merges(self):
        """Already one cluster — no adjacent pairs, zero merges."""
        bs = make_bin_scheme(["A-A"], n_bins=4)
        edges = [EdgeRecord(i=0, j=1, pair_key="A-A", pair_type_idx=0,
                            raw_value=2.0, bin_idx=1, cut_cost=2.0)]
        labels = np.array([0, 0], dtype=int)
        p = partition_from_labels(labels, edges, bs)
        n = cluster_merge_sweep(p)
        assert n == 0


# ---------------------------------------------------------------------------
# louvain_optimize — integration
# ---------------------------------------------------------------------------

class TestLouvainOptimize:
    def test_returns_result(self):
        p = _two_identical_clusters()
        from graincluster.optimizer.louvain import LouvainResult
        result = louvain_optimize(p)
        assert isinstance(result, LouvainResult)

    def test_objective_does_not_increase(self):
        p = _two_identical_clusters()
        result = louvain_optimize(p)
        assert result.objective_final <= result.objective_initial + 1e-9

    def test_escapes_atom_level_local_minimum(self):
        """Cluster-merge phase escapes a local minimum that atom moves cannot.

        Two identical clusters connected by one bridge edge.
        beta=0.9: cut penalty dominates single-atom moves (4 net cuts × 2 nats),
        so all atom moves are blocked (verified: greedy makes 0 moves).
        cluster_merge ΔL = (1-β)*ΔL_data - β*cut_savings - gamma < 0,
        so the cluster-merge phase merges them in one sweep.
        """
        from graincluster.optimizer.greedy import greedy_optimize

        p_greedy = _identical_clusters_cut_blocked()
        p_louvain = _identical_clusters_cut_blocked()

        result_greedy = greedy_optimize(p_greedy, max_passes=50)
        assert result_greedy.n_moves == 0        # confirm greedy is stuck
        assert p_greedy.n_clusters() == 2

        louvain_optimize(p_louvain, max_rounds=10)
        assert p_louvain.n_clusters() == 1       # Louvain escapes

    def test_two_phases_reported(self):
        """Louvain reports cluster merges when atom sweep alone is stuck."""
        p = _identical_clusters_cut_blocked()
        result = louvain_optimize(p)
        assert result.n_cluster_merges >= 1

    def test_distinct_phases_stay_separate(self):
        """Entropy-dominated regime: Louvain does not merge distinct clusters."""
        p = _two_distinct_clusters()
        louvain_optimize(p)
        left_label = int(p.atom_labels[0])
        right_label = int(p.atom_labels[-1])
        assert left_label != right_label

    def test_singleton_init_converges(self):
        """All-singleton start → Louvain finds compact solution."""
        n = 5
        bs = make_bin_scheme(["A-A"], n_bins=10, lo=1.0, hi=5.0)
        edges = make_two_domain_edges(
            n_a=n, n_b=n,
            pair_key="A-A", pair_type_idx=0,
            raw_a=2.0, raw_b=4.0, bridge_raw=3.0,
            bin_scheme=bs,
        )
        labels = np.arange(2 * n, dtype=int)
        p = partition_from_labels(labels, edges, bs, alpha=0.5, gamma=0.5, beta=0.5)
        result = louvain_optimize(p)
        assert result.objective_final < result.objective_initial
        assert p.n_clusters() < 2 * n
