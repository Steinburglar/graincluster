"""Partition: full clustering state for one frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..graph.edge import EdgeRecord
from ..features.binning import BinScheme
from .cluster import ClusterState
from .entropy import cluster_entropy, data_term, data_term_from_counts, self_information


@dataclass
class Partition:
    """Complete clustering state: atom assignments + cluster count tables.

    atom_labels[i] = cluster_id for atom i.
    clusters maps cluster_id -> ClusterState.
    Edges are stored once; internal/cut status derived from atom_labels.
    """

    atom_labels: np.ndarray            # shape (n_atoms,), int
    clusters: dict[int, ClusterState]
    edges: list[EdgeRecord]
    bin_scheme: BinScheme
    alpha: float = 0.5
    gamma: float = 1.0
    beta: float = 0.5   # cut-entropy balance ∈ [0,1]; 0=pure entropy, 1=pure cut
    _M: int = field(init=False, repr=False)
    _next_cluster_id: int = field(init=False, repr=False)
    # adjacency index: atom -> list of edge indices
    _adj: dict[int, list[int]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._M = self.bin_scheme.total_categories()
        self._next_cluster_id = max(self.clusters.keys(), default=-1) + 1
        self._build_adj()

    def _build_adj(self) -> None:
        self._adj = {i: [] for i in range(len(self.atom_labels))}
        for eidx, e in enumerate(self.edges):
            self._adj[e.i].append(eidx)
            self._adj[e.j].append(eidx)

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def objective(self) -> float:
        """Full L = (1-β) Σ_C N_C H_C + β Σ_cut s_ij + γ K."""
        L_data = (1.0 - self.beta) * sum(
            data_term(c, self._M, self.alpha) for c in self.clusters.values()
        )
        K = len(self.clusters)
        L_cut = self.beta * sum(
            e.cut_cost
            for e in self.edges
            if self.atom_labels[e.i] != self.atom_labels[e.j]
        )
        return L_data + self.gamma * K + L_cut

    # ------------------------------------------------------------------
    # Move scoring (frozen model)
    # ------------------------------------------------------------------

    def score_move(
        self,
        atom: int,
        target_cluster_id: int,
        exact_below_N: int = 0,
    ) -> float:
        """ΔL for moving atom from its current cluster to target_cluster_id.

        Does NOT mutate partition.

        exact_below_N: if > 0, use exact (non-frozen) data-term delta when
        src.N < exact_below_N or tgt.N < exact_below_N.  This corrects the
        frozen-model underestimate of entropy gain when removing the last few
        edges from a small cluster.  exact_below_N=0 (default) uses frozen
        model everywhere, matching the original behaviour.
        """
        src_id = int(self.atom_labels[atom])
        if src_id == target_cluster_id:
            return 0.0

        src = self.clusters[src_id]
        tgt_is_new = target_cluster_id not in self.clusters
        tgt = self.clusters.get(target_cluster_id)

        M = self._M
        alpha = self.alpha

        # --- cut cost delta (same for both frozen and exact paths) ---
        delta_cut = 0.0
        for eidx in self._adj[atom]:
            e = self.edges[eidx]
            neighbor = e.j if e.i == atom else e.i
            nbr_cluster = int(self.atom_labels[neighbor])
            if nbr_cluster == src_id:
                delta_cut += self.beta * e.cut_cost
            elif nbr_cluster == target_cluster_id:
                delta_cut -= self.beta * e.cut_cost

        # --- entropy / data-term delta ---
        use_exact = (
            exact_below_N > 0
            and (src.N < exact_below_N or (tgt is not None and tgt.N < exact_below_N))
        )

        if use_exact:
            delta_entropy = self._exact_data_delta(atom, src_id, target_cluster_id)
        else:
            delta_entropy = 0.0
            for eidx in self._adj[atom]:
                e = self.edges[eidx]
                neighbor = e.j if e.i == atom else e.i
                nbr_cluster = int(self.atom_labels[neighbor])
                if nbr_cluster == src_id:
                    delta_entropy -= self_information(
                        e.pair_type_idx, e.bin_idx, src, M, alpha
                    )
                elif nbr_cluster == target_cluster_id:
                    if tgt is not None:
                        delta_entropy += self_information(
                            e.pair_type_idx, e.bin_idx, tgt, M, alpha
                        )
                    else:
                        import math
                        delta_entropy += math.log(M) if M > 0 else 0.0

        # --- cluster count delta ---
        src_becomes_empty = len(src.atom_ids) == 1
        delta_K = 0
        if tgt_is_new:
            delta_K += 1
        if src_becomes_empty:
            delta_K -= 1

        return (1.0 - self.beta) * delta_entropy + delta_cut + self.gamma * delta_K

    def _exact_data_delta(
        self,
        atom: int,
        src_id: int,
        target_cluster_id: int,
    ) -> float:
        """Exact Δ(N_src*H_src + N_tgt*H_tgt) without frozen-model approximation.

        Constructs post-move count tables by walking incident edges, then
        recomputes data terms exactly.  Called only when src or tgt is small.
        """
        M = self._M
        alpha = self.alpha
        src = self.clusters[src_id]
        tgt = self.clusters.get(target_cluster_id)

        src_counts: dict = dict(src.counts)
        src_N: int = src.N
        tgt_counts: dict = dict(tgt.counts) if tgt is not None else {}
        tgt_N: int = tgt.N if tgt is not None else 0

        for eidx in self._adj[atom]:
            e = self.edges[eidx]
            neighbor = e.j if e.i == atom else e.i
            nbr_cid = int(self.atom_labels[neighbor])
            key = (e.pair_type_idx, e.bin_idx)
            if nbr_cid == src_id:
                c = src_counts.get(key, 0) - 1
                if c <= 0:
                    src_counts.pop(key, None)
                else:
                    src_counts[key] = c
                src_N -= 1
            elif nbr_cid == target_cluster_id and tgt is not None:
                tgt_counts[key] = tgt_counts.get(key, 0) + 1
                tgt_N += 1

        dt_before = data_term(src, M, alpha) + (data_term(tgt, M, alpha) if tgt is not None else 0.0)
        dt_after = data_term_from_counts(src_counts, src_N, M, alpha) + data_term_from_counts(tgt_counts, tgt_N, M, alpha)
        return dt_after - dt_before

    # ------------------------------------------------------------------
    # Move application
    # ------------------------------------------------------------------

    def apply_move(self, atom: int, target_cluster_id: int) -> None:
        """Move atom to target_cluster_id, updating all state."""
        src_id = int(self.atom_labels[atom])
        if src_id == target_cluster_id:
            return

        src = self.clusters[src_id]
        tgt_is_new = target_cluster_id not in self.clusters

        if tgt_is_new:
            tgt = ClusterState(cluster_id=target_cluster_id)
            self.clusters[target_cluster_id] = tgt
            if target_cluster_id >= self._next_cluster_id:
                self._next_cluster_id = target_cluster_id + 1
        else:
            tgt = self.clusters[target_cluster_id]

        # Update atom assignment.
        self.atom_labels[atom] = target_cluster_id
        src.atom_ids.discard(atom)
        tgt.atom_ids.add(atom)

        # Update edge counts.
        for eidx in self._adj[atom]:
            e = self.edges[eidx]
            neighbor = e.j if e.i == atom else e.i
            nbr_cluster = int(self.atom_labels[neighbor])

            if nbr_cluster == src_id:
                # Was internal to src, now cut. Remove from src.
                src.remove_edge(e.pair_type_idx, e.bin_idx)

            elif nbr_cluster == target_cluster_id:
                # Was cut, now internal to tgt. Add to tgt.
                tgt.add_edge(e.pair_type_idx, e.bin_idx)

        # Remove src if empty.
        if src.is_empty():
            del self.clusters[src_id]

    def new_cluster_id(self) -> int:
        """Return an unused cluster id (does not create the cluster)."""
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        return cid

    # ------------------------------------------------------------------
    # Cluster-level merge (Louvain aggregation phase)
    # ------------------------------------------------------------------

    def score_cluster_merge(self, cid_a: int, cid_b: int) -> float:
        """Exact ΔL for merging cluster cid_a into cid_b as one compound move.

        Scans the smaller cluster's adjacency to find cross edges once.
        Does NOT mutate partition.
        """
        ca = self.clusters[cid_a]
        cb = self.clusters[cid_b]
        M = self._M
        alpha = self.alpha

        # Scan atoms of the smaller cluster for cross edges to the other.
        if len(ca.atom_ids) <= len(cb.atom_ids):
            scan_ids, other_cid = ca.atom_ids, cid_b
        else:
            scan_ids, other_cid = cb.atom_ids, cid_a

        between_counts: dict = {}
        between_N: int = 0
        between_cut_cost: float = 0.0

        for atom in scan_ids:
            for eidx in self._adj[atom]:
                e = self.edges[eidx]
                nbr = e.j if e.i == atom else e.i
                if int(self.atom_labels[nbr]) == other_cid:
                    key = (e.pair_type_idx, e.bin_idx)
                    between_counts[key] = between_counts.get(key, 0) + 1
                    between_N += 1
                    between_cut_cost += e.cut_cost

        merged_counts: dict = dict(ca.counts)
        for k, v in cb.counts.items():
            merged_counts[k] = merged_counts.get(k, 0) + v
        for k, v in between_counts.items():
            merged_counts[k] = merged_counts.get(k, 0) + v
        merged_N = ca.N + cb.N + between_N

        dt_merged = data_term_from_counts(merged_counts, merged_N, M, alpha)
        dt_before = data_term(ca, M, alpha) + data_term(cb, M, alpha)

        return (1.0 - self.beta) * (dt_merged - dt_before) - self.beta * between_cut_cost - self.gamma

    def apply_cluster_merge(self, src_cid: int, tgt_cid: int) -> None:
        """Absorb cluster src_cid into tgt_cid, updating all state.

        src_cid is deleted; tgt_cid accumulates all atoms, counts, and
        the formerly-cut edges between them become internal.
        """
        src = self.clusters[src_cid]
        tgt = self.clusters[tgt_cid]

        # Cross edges (src atom ↔ tgt atom): become internal to tgt.
        for atom in src.atom_ids:
            for eidx in self._adj[atom]:
                e = self.edges[eidx]
                nbr = e.j if e.i == atom else e.i
                if int(self.atom_labels[nbr]) == tgt_cid:
                    tgt.counts[(e.pair_type_idx, e.bin_idx)] = (
                        tgt.counts.get((e.pair_type_idx, e.bin_idx), 0) + 1
                    )
                    tgt.N += 1
        tgt._entropy = None

        # Transfer src internal edges to tgt.
        for k, v in src.counts.items():
            tgt.counts[k] = tgt.counts.get(k, 0) + v
        tgt.N += src.N
        tgt._entropy = None

        # Reassign atoms.
        for atom in src.atom_ids:
            self.atom_labels[atom] = tgt_cid
        tgt.atom_ids |= src.atom_ids

        del self.clusters[src_cid]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def n_clusters(self) -> int:
        return len(self.clusters)

    def entropy_per_cluster(self) -> dict[int, float]:
        return {cid: cluster_entropy(c, self._M, self.alpha)
                for cid, c in self.clusters.items()}


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def partition_from_labels(
    atom_labels: np.ndarray,
    edges: list[EdgeRecord],
    bin_scheme: BinScheme,
    alpha: float = 0.5,
    gamma: float = 1.0,
    beta: float = 0.5,
) -> Partition:
    """Build a Partition by scanning edges to populate cluster count tables."""
    atom_labels = np.asarray(atom_labels, dtype=int)
    n_atoms = len(atom_labels)

    clusters: dict[int, ClusterState] = {}
    for atom_idx in range(n_atoms):
        cid = int(atom_labels[atom_idx])
        if cid not in clusters:
            clusters[cid] = ClusterState(cluster_id=cid)
        clusters[cid].atom_ids.add(atom_idx)

    for e in edges:
        ci = int(atom_labels[e.i])
        cj = int(atom_labels[e.j])
        if ci == cj:
            clusters[ci].add_edge(e.pair_type_idx, e.bin_idx)

    return Partition(
        atom_labels=atom_labels,
        clusters=clusters,
        edges=edges,
        bin_scheme=bin_scheme,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
    )
