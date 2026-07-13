"""Partition: full clustering state for one frame."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..graph.edge import EdgeRecord
from ..features.binning import BinScheme
from .cluster import ClusterState
from .entropy import cluster_entropy
from .parameterized import (
    cluster_count_prior_predictive_cost,
    cluster_data_term,
    edge_data_term_for_pair_type,
    lomax_cut_prior_cost,
    parameterized_multinomial_cost,
    species_data_term,
    uniform_alpha,
)

# Sentinel cluster ID for the permanent "other" / background cluster.
# Always exists, never deleted, excluded from gamma*K.
OTHER_ID: int = -1


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
    cluster_count_prior_mean: float | None = None
    cluster_count_prior_strength: float | None = None
    cluster_count_prior_tau: float | None = None
    cut_prior_beta0: float | None = None
    atom_species_idx: np.ndarray | None = None
    kappa_species: float | None = None
    kappa_edge: float | None = None
    alpha_species: np.ndarray | None = None
    alpha_edge_by_type: dict[int, np.ndarray] | None = None
    parameter_estimator: str = "constrained_map"
    estimator_epsilon: float = 1e-12
    _M: int = field(init=False, repr=False)
    _next_cluster_id: int = field(init=False, repr=False)
    # adjacency index: atom -> list of edge indices
    _adj: dict[int, list[int]] = field(init=False, repr=False)
    _cut_cost_total: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._M = self.bin_scheme.total_categories()
        if self.atom_species_idx is None:
            self.atom_species_idx = np.zeros(len(self.atom_labels), dtype=int)
        else:
            self.atom_species_idx = np.asarray(self.atom_species_idx, dtype=int)
        if len(self.atom_species_idx) != len(self.atom_labels):
            raise ValueError("atom_species_idx must have one entry per atom")
        self._init_priors()
        # Exclude OTHER_ID from the ID counter; real cluster IDs start at 0.
        real_ids = [k for k in self.clusters if k != OTHER_ID]
        self._next_cluster_id = max(real_ids, default=-1) + 1
        # Ensure the "other" cluster always exists.
        if OTHER_ID not in self.clusters:
            self.clusters[OTHER_ID] = ClusterState(cluster_id=OTHER_ID)
        self._ensure_species_counts()
        self._build_adj()
        self._cut_cost_total = self._compute_cut_cost_total()

    def _init_priors(self) -> None:
        n_species = int(self.atom_species_idx.max()) + 1 if len(self.atom_species_idx) else 1
        if self.alpha_species is None:
            species_counts = np.bincount(self.atom_species_idx, minlength=n_species).astype(float)
            if self.kappa_species is None:
                self.kappa_species = self.alpha * n_species
            self.alpha_species = self.kappa_species * species_counts / species_counts.sum()
        else:
            self.alpha_species = np.asarray(self.alpha_species, dtype=float)

        if self.alpha_edge_by_type is None:
            self.alpha_edge_by_type = {}
            for pair_type_idx, n_bins in self.bin_scheme.n_bins_per_type.items():
                if self.kappa_edge is None:
                    self.alpha_edge_by_type[pair_type_idx] = np.full(n_bins, self.alpha, dtype=float)
                else:
                    self.alpha_edge_by_type[pair_type_idx] = uniform_alpha(n_bins, self.kappa_edge)
        else:
            self.alpha_edge_by_type = {
                int(k): np.asarray(v, dtype=float)
                for k, v in self.alpha_edge_by_type.items()
            }

    def _ensure_species_counts(self) -> None:
        for cluster in self.clusters.values():
            if cluster.species_counts:
                continue
            for atom in cluster.atom_ids:
                cluster.add_atom_species(int(self.atom_species_idx[atom]))

    def _build_adj(self) -> None:
        self._adj = {i: [] for i in range(len(self.atom_labels))}
        for eidx, e in enumerate(self.edges):
            self._adj[e.i].append(eidx)
            self._adj[e.j].append(eidx)

    def _compute_cut_cost_total(self) -> float:
        return sum(
            e.cut_cost
            for e in self.edges
            if self.atom_labels[e.i] != self.atom_labels[e.j]
        )

    def _using_new_prior_framework(self) -> bool:
        return (
            self.cluster_count_prior_tau is not None
            or self.cluster_count_prior_strength is not None
            or self.cut_prior_beta0 is not None
        )

    def _cluster_count_prior_active(self) -> bool:
        return (
            self.cluster_count_prior_tau is not None
            or self.cluster_count_prior_strength is not None
        )

    def _cut_lomax_prior_active(self) -> bool:
        return self.cut_prior_beta0 is not None

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------

    def objective(self) -> float:
        """Full parameterized objective.

        OTHER_ID is excluded from the real cluster count K.
        """
        if self._using_new_prior_framework():
            L_data = sum(self._cluster_data_term(c) for c in self.clusters.values())
            return L_data + self._structure_prior_cost()

        L_data = (1.0 - self.beta) * sum(
            self._cluster_data_term(c) for c in self.clusters.values()
        )
        return L_data + self.beta * self._cut_cost_total + self.gamma * self.n_clusters()

    def _cluster_data_term(self, cluster: ClusterState) -> float:
        return cluster_data_term(
            cluster,
            self.alpha_species,
            self.alpha_edge_by_type,
            estimator=self.parameter_estimator,
            epsilon=self.estimator_epsilon,
        )

    def _structure_prior_cost(self) -> float:
        result = 0.0
        if self._cluster_count_prior_active():
            result += self._cluster_count_prior_cost(self.n_clusters())
        if self._cut_lomax_prior_active():
            result += self._cut_lomax_prior_cost(self._cut_cost_total, self.n_clusters())
        return result

    def _cluster_count_prior_cost(self, k_clusters: int) -> float:
        mean = self.cluster_count_prior_mean
        strength = self.cluster_count_prior_strength
        if mean is None:
            mean = max(1.0, np.sqrt(len(self.atom_labels)))
        if strength is None:
            tau = 0.0 if self.cluster_count_prior_tau is None else float(self.cluster_count_prior_tau)
            strength = float(len(self.atom_labels)) * (10.0 ** tau)
        return cluster_count_prior_predictive_cost(
            k_clusters=k_clusters,
            mean=float(mean),
            strength=float(strength),
        )

    def _cut_lomax_prior_cost(self, cut_value: float, k_clusters: int) -> float:
        if self.cut_prior_beta0 is None:
            return 0.0
        return lomax_cut_prior_cost(
            cut_value=cut_value,
            n_atoms=len(self.atom_labels),
            k_clusters=k_clusters,
            beta0=float(self.cut_prior_beta0),
        )

    # ------------------------------------------------------------------
    # Move scoring
    # ------------------------------------------------------------------

    def score_move(
        self,
        atom: int,
        target_cluster_id: int,
    ) -> float:
        """Exact ΔL for moving atom from current cluster to target_cluster_id.

        Does NOT mutate partition.
        """
        src_id = int(self.atom_labels[atom])
        if src_id == target_cluster_id:
            return 0.0

        src = self.clusters[src_id]
        tgt_is_new = target_cluster_id not in self.clusters

        source_after = self._source_after_removal_state(atom, src_id)
        split_occurs = src_id != OTHER_ID and len(source_after) > 1
        if split_occurs:
            delta_data = self._exact_data_delta_with_split(
                atom,
                src_id,
                target_cluster_id,
                source_after,
            )
        else:
            delta_data = self._exact_data_delta(atom, src_id, target_cluster_id)

        # --- cluster count delta ---
        # OTHER_ID never counts toward K (free background), so:
        #   - src OTHER_ID emptying: no delta_K (other persists, was never in K)
        #   - tgt OTHER_ID: never "new" (always exists, not in K), no delta_K
        src_becomes_empty = len(source_after) == 0 and src_id != OTHER_ID
        delta_K = 0
        if tgt_is_new:
            delta_K += 1
        if src_becomes_empty:
            delta_K -= 1
        elif src_id != OTHER_ID and len(source_after) > 1:
            delta_K += len(source_after) - 1

        delta_cut_raw = 0.0
        if not self._using_new_prior_framework() or self._cut_lomax_prior_active():
            delta_cut_raw = self._cut_cost_delta_for_move(
                atom,
                src_id,
                target_cluster_id,
                source_after=source_after,
            )

        delta_structure = self._structure_prior_delta(delta_cut_raw, delta_K)
        if self._using_new_prior_framework():
            return delta_data + delta_structure
        return (1.0 - self.beta) * delta_data + delta_structure + self.gamma * delta_K

    def _structure_prior_delta(
        self,
        delta_cut_raw: float,
        delta_k: int,
    ) -> float:
        if self._using_new_prior_framework():
            k_before = self.n_clusters()
            k_after = k_before + delta_k
            result = 0.0
            if self._cluster_count_prior_active():
                result += (
                    self._cluster_count_prior_cost(k_after)
                    - self._cluster_count_prior_cost(k_before)
                )
            if self._cut_lomax_prior_active():
                result += (
                    self._cut_lomax_prior_cost(self._cut_cost_total + delta_cut_raw, k_after)
                    - self._cut_lomax_prior_cost(self._cut_cost_total, k_before)
                )
            return result
        return self.beta * delta_cut_raw

    def _cut_cost_delta_for_move(
        self,
        atom: int,
        src_id: int,
        target_cluster_id: int,
        source_after: list[tuple[dict[int, int], dict[tuple[int, int], int], set[int]]] | None = None,
    ) -> float:
        delta_cut = 0.0
        for eidx in self._adj[atom]:
            e = self.edges[eidx]
            neighbor = e.j if e.i == atom else e.i
            nbr_cluster = int(self.atom_labels[neighbor])
            if nbr_cluster == src_id:
                delta_cut += e.cut_cost
            elif nbr_cluster == target_cluster_id:
                delta_cut -= e.cut_cost
        if source_after is not None and len(source_after) > 1:
            comp_idx: dict[int, int] = {}
            for idx, (_, _, atom_ids) in enumerate(source_after):
                for member in atom_ids:
                    comp_idx[member] = idx
            remaining_atoms = set(comp_idx)
            seen_eidx: set[int] = set()
            for member in remaining_atoms:
                for eidx in self._adj[member]:
                    if eidx in seen_eidx:
                        continue
                    seen_eidx.add(eidx)
                    e = self.edges[eidx]
                    if e.i not in remaining_atoms or e.j not in remaining_atoms:
                        continue
                    if comp_idx[e.i] != comp_idx[e.j]:
                        delta_cut += e.cut_cost
        return delta_cut

    def _exact_data_delta(
        self,
        atom: int,
        src_id: int,
        target_cluster_id: int,
    ) -> float:
        """Exact parameterized data-term delta for one atom move without source split."""
        src = self.clusters[src_id]
        tgt = self.clusters.get(target_cluster_id)
        src_after_species = dict(src.species_counts)
        tgt_after_species = dict(tgt.species_counts) if tgt is not None else {}
        species_idx = int(self.atom_species_idx[atom])
        self._decrement_count(src_after_species, species_idx)
        tgt_after_species[species_idx] = tgt_after_species.get(species_idx, 0) + 1

        touched_pair_types: set[int] = set()
        src_edge_by_type: dict[int, dict[int, int]] = {}
        tgt_edge_by_type: dict[int, dict[int, int]] = {}
        for eidx in self._adj[atom]:
            e = self.edges[eidx]
            neighbor = e.j if e.i == atom else e.i
            nbr_cid = int(self.atom_labels[neighbor])
            pt_idx = e.pair_type_idx
            if nbr_cid == src_id:
                touched_pair_types.add(pt_idx)
                src_edge_by_type.setdefault(pt_idx, src.edge_counts_for_pair_type(pt_idx))
                self._decrement_count(src_edge_by_type[pt_idx], e.bin_idx)
            elif nbr_cid == target_cluster_id and tgt is not None:
                touched_pair_types.add(pt_idx)
                tgt_edge_by_type.setdefault(pt_idx, tgt.edge_counts_for_pair_type(pt_idx))
                tgt_edge_by_type[pt_idx][e.bin_idx] = tgt_edge_by_type[pt_idx].get(e.bin_idx, 0) + 1

        before = species_data_term(
            src.species_counts,
            self.alpha_species,
            estimator=self.parameter_estimator,
            epsilon=self.estimator_epsilon,
        )
        after = species_data_term(
            src_after_species,
            self.alpha_species,
            estimator=self.parameter_estimator,
            epsilon=self.estimator_epsilon,
        )
        if tgt is not None:
            before += species_data_term(
                tgt.species_counts,
                self.alpha_species,
                estimator=self.parameter_estimator,
                epsilon=self.estimator_epsilon,
            )
            after += species_data_term(
                tgt_after_species,
                self.alpha_species,
                estimator=self.parameter_estimator,
                epsilon=self.estimator_epsilon,
            )
        else:
            after += species_data_term(
                tgt_after_species,
                self.alpha_species,
                estimator=self.parameter_estimator,
                epsilon=self.estimator_epsilon,
            )

        for pt_idx in touched_pair_types:
            alpha_edge = self.alpha_edge_by_type[pt_idx]
            src_before_counts = src.edge_counts_for_pair_type(pt_idx)
            before += edge_data_term_for_pair_type(
                src_before_counts,
                alpha_edge,
                estimator=self.parameter_estimator,
                epsilon=self.estimator_epsilon,
            )
            after += edge_data_term_for_pair_type(
                src_edge_by_type.get(pt_idx, src_before_counts),
                alpha_edge,
                estimator=self.parameter_estimator,
                epsilon=self.estimator_epsilon,
            )
            if tgt is not None:
                tgt_before_counts = tgt.edge_counts_for_pair_type(pt_idx)
                before += edge_data_term_for_pair_type(
                    tgt_before_counts,
                    alpha_edge,
                    estimator=self.parameter_estimator,
                    epsilon=self.estimator_epsilon,
                )
                after += edge_data_term_for_pair_type(
                    tgt_edge_by_type.get(pt_idx, tgt_before_counts),
                    alpha_edge,
                    estimator=self.parameter_estimator,
                    epsilon=self.estimator_epsilon,
                )

        return after - before

    def _exact_data_delta_with_split(
        self,
        atom: int,
        src_id: int,
        target_cluster_id: int,
        source_after: list[tuple[dict[int, int], dict[tuple[int, int], int], set[int]]],
    ) -> float:
        """Exact data delta when removing one atom disconnects the source cluster."""
        src = self.clusters[src_id]
        tgt = self.clusters.get(target_cluster_id)
        tgt_species_counts, tgt_counts = self._target_after_parts(atom, src_id, target_cluster_id)

        before = self._cluster_data_term(src) + (self._cluster_data_term(tgt) if tgt is not None else 0.0)
        after = 0.0
        for species_counts, edge_counts, _atom_ids in source_after:
            after += self._data_term_from_parts(species_counts, edge_counts)
        after += self._data_term_from_parts(tgt_species_counts, tgt_counts)
        return after - before

    @staticmethod
    def _decrement_count(counts: dict, key) -> None:
        c = counts.get(key, 0) - 1
        if c <= 0:
            counts.pop(key, None)
        else:
            counts[key] = c

    def _data_term_from_parts(
        self,
        species_counts: dict[int, int],
        edge_counts: dict[tuple[int, int], int],
    ) -> float:
        result = parameterized_multinomial_cost(
            self._dense_species_counts(species_counts),
            self.alpha_species,
            estimator=self.parameter_estimator,
            epsilon=self.estimator_epsilon,
        ) if sum(species_counts.values()) > 0 else 0.0
        for pair_type_idx, alpha_edge in self.alpha_edge_by_type.items():
            dense = np.zeros(len(alpha_edge), dtype=float)
            touched = False
            for (pt_idx, bin_idx), count in edge_counts.items():
                if pt_idx == pair_type_idx:
                    dense[bin_idx] = count
                    touched = True
            if touched:
                result += parameterized_multinomial_cost(
                    dense,
                    alpha_edge,
                    estimator=self.parameter_estimator,
                    epsilon=self.estimator_epsilon,
                )
        return result

    def _target_after_parts(
        self,
        atom: int,
        src_id: int,
        target_cluster_id: int,
    ) -> tuple[dict[int, int], dict[tuple[int, int], int]]:
        tgt = self.clusters.get(target_cluster_id)
        tgt_counts: dict[tuple[int, int], int] = dict(tgt.counts) if tgt is not None else {}
        tgt_species_counts: dict[int, int] = dict(tgt.species_counts) if tgt is not None else {}

        species_idx = int(self.atom_species_idx[atom])
        tgt_species_counts[species_idx] = tgt_species_counts.get(species_idx, 0) + 1

        for eidx in self._adj[atom]:
            e = self.edges[eidx]
            neighbor = e.j if e.i == atom else e.i
            nbr_cid = int(self.atom_labels[neighbor])
            if nbr_cid == target_cluster_id and target_cluster_id in self.clusters:
                key = (e.pair_type_idx, e.bin_idx)
                tgt_counts[key] = tgt_counts.get(key, 0) + 1
        return tgt_species_counts, tgt_counts

    def _source_after_removal_state(
        self,
        atom: int,
        src_id: int,
    ) -> list[tuple[dict[int, int], dict[tuple[int, int], int], set[int]]]:
        """Return source connected components after removing one atom.

        Each component is returned as:

        - species count dict
        - edge count dict
        - atom id set
        """
        if src_id == OTHER_ID:
            return []

        src = self.clusters[src_id]
        remaining = set(src.atom_ids)
        remaining.discard(atom)
        if not remaining:
            return []

        remaining_set = remaining
        components: list[set[int]] = []
        atom_to_comp: dict[int, int] = {}
        unseen = set(remaining)
        while unseen:
            start = next(iter(unseen))
            comp: set[int] = set()
            stack = [start]
            while stack:
                member = stack.pop()
                if member in comp:
                    continue
                comp.add(member)
                for eidx in self._adj[member]:
                    e = self.edges[eidx]
                    nbr = e.j if e.i == member else e.i
                    if nbr in remaining_set and nbr not in comp:
                        stack.append(nbr)
            comp_idx = len(components)
            for member in comp:
                atom_to_comp[member] = comp_idx
            components.append(comp)
            unseen -= comp

        component_species = [dict() for _ in components]
        component_edges = [dict() for _ in components]
        for member, comp_idx in atom_to_comp.items():
            species_idx = int(self.atom_species_idx[member])
            component_species[comp_idx][species_idx] = component_species[comp_idx].get(species_idx, 0) + 1

        seen_eidx: set[int] = set()
        for member in remaining_set:
            comp_idx = atom_to_comp[member]
            for eidx in self._adj[member]:
                if eidx in seen_eidx:
                    continue
                seen_eidx.add(eidx)
                e = self.edges[eidx]
                if e.i not in remaining_set or e.j not in remaining_set:
                    continue
                other_comp_idx = atom_to_comp[e.j if e.i == member else e.i]
                if comp_idx != other_comp_idx:
                    continue
                key = (e.pair_type_idx, e.bin_idx)
                component_edges[comp_idx][key] = component_edges[comp_idx].get(key, 0) + 1

        return [
            (species_counts, edge_counts, atom_ids)
            for species_counts, edge_counts, atom_ids in zip(component_species, component_edges, components)
        ]

    def _dense_species_counts(self, species_counts: dict[int, int]) -> np.ndarray:
        dense = np.zeros(len(self.alpha_species), dtype=float)
        for species_idx, count in species_counts.items():
            dense[species_idx] = count
        return dense

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
        delta_cut_raw = self._cut_cost_delta_for_move(atom, src_id, target_cluster_id)

        # Update atom assignment.
        self.atom_labels[atom] = target_cluster_id
        src.atom_ids.discard(atom)
        tgt.atom_ids.add(atom)
        species_idx = int(self.atom_species_idx[atom])
        src.remove_atom_species(species_idx)
        tgt.add_atom_species(species_idx)

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

        # Remove src if empty (OTHER_ID is never deleted).
        if src.is_empty() and src_id != OTHER_ID:
            del self.clusters[src_id]
        self._cut_cost_total += delta_cut_raw
        if src_id != OTHER_ID and src_id in self.clusters:
            self._split_cluster_if_disconnected(src_id)

    def new_cluster_id(self) -> int:
        """Return an unused cluster id (does not create the cluster)."""
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        return cid

    def _split_cluster_if_disconnected(self, cluster_id: int) -> int:
        """Split cluster into connected components if needed.

        Smaller components become new clusters; the largest component keeps the
        original id. Returns the number of new clusters created.
        """
        c = self.clusters.get(cluster_id)
        if c is None or len(c.atom_ids) <= 1:
            return 0

        remaining = set(c.atom_ids)
        components: list[set[int]] = []
        while remaining:
            start = next(iter(remaining))
            comp: set[int] = set()
            stack = [start]
            while stack:
                atom = stack.pop()
                if atom in comp:
                    continue
                comp.add(atom)
                for eidx in self._adj[atom]:
                    e = self.edges[eidx]
                    nbr = e.j if e.i == atom else e.i
                    if int(self.atom_labels[nbr]) == cluster_id and nbr not in comp:
                        stack.append(nbr)
            components.append(comp)
            remaining -= comp

        if len(components) == 1:
            return 0

        largest = max(components, key=len)
        n_new = 0
        delta_cut = 0.0
        for small_comp in components:
            if small_comp is largest:
                continue

            new_cid = self.new_cluster_id()
            new_c = ClusterState(cluster_id=new_cid)
            self.clusters[new_cid] = new_c
            new_c.atom_ids = small_comp.copy()

            for atom in small_comp:
                self.atom_labels[atom] = new_cid
                c.atom_ids.discard(atom)
                species_idx = int(self.atom_species_idx[atom])
                c.remove_atom_species(species_idx)
                new_c.add_atom_species(species_idx)

            seen_eidx: set[int] = set()
            for atom in small_comp:
                for eidx in self._adj[atom]:
                    if eidx in seen_eidx:
                        continue
                    seen_eidx.add(eidx)
                    e = self.edges[eidx]
                    nbr = e.j if e.i == atom else e.i
                    nbr_label = int(self.atom_labels[nbr])
                    key = (e.pair_type_idx, e.bin_idx)

                    if nbr in small_comp:
                        new_c.counts[key] = new_c.counts.get(key, 0) + 1
                        new_c.N += 1
                        cnt = c.counts.get(key, 0) - 1
                        if cnt <= 0:
                            c.counts.pop(key, None)
                        else:
                            c.counts[key] = cnt
                        c.N -= 1
                    elif nbr_label == cluster_id:
                        cnt = c.counts.get(key, 0) - 1
                        if cnt <= 0:
                            c.counts.pop(key, None)
                        else:
                            c.counts[key] = cnt
                        c.N -= 1
                        delta_cut += e.cut_cost

            c.invalidate_entropy()
            new_c.invalidate_entropy()
            n_new += 1

        c.invalidate_entropy()
        self._cut_cost_total += delta_cut
        return n_new

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

        # Scan atoms of the smaller cluster for cross edges to the other.
        if len(ca.atom_ids) <= len(cb.atom_ids):
            scan_ids, other_cid = ca.atom_ids, cid_b
        else:
            scan_ids, other_cid = cb.atom_ids, cid_a

        between_counts: dict = {}
        between_cut_cost: float = 0.0

        for atom in scan_ids:
            for eidx in self._adj[atom]:
                e = self.edges[eidx]
                nbr = e.j if e.i == atom else e.i
                if int(self.atom_labels[nbr]) == other_cid:
                    key = (e.pair_type_idx, e.bin_idx)
                    between_counts[key] = between_counts.get(key, 0) + 1
                    between_cut_cost += e.cut_cost

        merged_counts: dict = dict(ca.counts)
        for k, v in cb.counts.items():
            merged_counts[k] = merged_counts.get(k, 0) + v
        for k, v in between_counts.items():
            merged_counts[k] = merged_counts.get(k, 0) + v

        merged_species_counts: dict = dict(ca.species_counts)
        for k, v in cb.species_counts.items():
            merged_species_counts[k] = merged_species_counts.get(k, 0) + v

        dt_merged = self._data_term_from_parts(merged_species_counts, merged_counts)
        dt_before = self._cluster_data_term(ca) + self._cluster_data_term(cb)

        delta_structure = self._structure_prior_delta_for_merge(between_cut_cost)
        if self._using_new_prior_framework():
            return (dt_merged - dt_before) + delta_structure
        return (1.0 - self.beta) * (dt_merged - dt_before) + delta_structure - self.gamma

    def _structure_prior_delta_for_merge(self, between_cut_cost: float) -> float:
        if self._using_new_prior_framework():
            k_before = self.n_clusters()
            k_after = k_before - 1
            result = 0.0
            if self._cluster_count_prior_active():
                result += (
                    self._cluster_count_prior_cost(k_after)
                    - self._cluster_count_prior_cost(k_before)
                )
            if self._cut_lomax_prior_active():
                result += (
                    self._cut_lomax_prior_cost(self._cut_cost_total - between_cut_cost, k_after)
                    - self._cut_lomax_prior_cost(self._cut_cost_total, k_before)
                )
            return result
        return -self.beta * between_cut_cost

    def apply_cluster_merge(self, src_cid: int, tgt_cid: int) -> None:
        """Absorb cluster src_cid into tgt_cid, updating all state.

        src_cid is deleted; tgt_cid accumulates all atoms, counts, and
        the formerly-cut edges between them become internal.
        """
        src = self.clusters[src_cid]
        tgt = self.clusters[tgt_cid]
        between_cut_cost = 0.0

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
                    between_cut_cost += e.cut_cost
        tgt._entropy = None

        # Transfer src internal edges to tgt.
        for k, v in src.counts.items():
            tgt.counts[k] = tgt.counts.get(k, 0) + v
        tgt.N += src.N
        tgt._entropy = None

        # Transfer species counts to tgt.
        for k, v in src.species_counts.items():
            tgt.species_counts[k] = tgt.species_counts.get(k, 0) + v

        # Reassign atoms.
        for atom in src.atom_ids:
            self.atom_labels[atom] = tgt_cid
        tgt.atom_ids |= src.atom_ids
        self._cut_cost_total -= between_cut_cost

        del self.clusters[src_cid]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def n_clusters(self) -> int:
        """Number of real clusters (excludes OTHER_ID)."""
        return sum(1 for cid in self.clusters if cid != OTHER_ID)

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
    atom_species: np.ndarray | list[int] | list[str] | None = None,
    alpha: float = 0.5,
    gamma: float = 1.0,
    beta: float = 0.5,
    cluster_count_prior_mean: float | None = None,
    cluster_count_prior_strength: float | None = None,
    cluster_count_prior_tau: float | None = None,
    cut_prior_beta0: float | None = None,
    kappa_species: float | None = None,
    kappa_edge: float | None = None,
    parameter_estimator: str = "constrained_map",
    estimator_epsilon: float = 1e-12,
) -> Partition:
    """Build a Partition by scanning edges to populate cluster count tables."""
    atom_labels = np.asarray(atom_labels, dtype=int)
    n_atoms = len(atom_labels)
    atom_species_idx = _coerce_atom_species(atom_species, n_atoms)

    clusters: dict[int, ClusterState] = {OTHER_ID: ClusterState(cluster_id=OTHER_ID)}
    for atom_idx in range(n_atoms):
        cid = int(atom_labels[atom_idx])
        if cid not in clusters:
            clusters[cid] = ClusterState(cluster_id=cid)
        clusters[cid].atom_ids.add(atom_idx)
        clusters[cid].add_atom_species(int(atom_species_idx[atom_idx]))

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
        cluster_count_prior_mean=cluster_count_prior_mean,
        cluster_count_prior_strength=cluster_count_prior_strength,
        cluster_count_prior_tau=cluster_count_prior_tau,
        cut_prior_beta0=cut_prior_beta0,
        atom_species_idx=atom_species_idx,
        kappa_species=kappa_species,
        kappa_edge=kappa_edge,
        parameter_estimator=parameter_estimator,
        estimator_epsilon=estimator_epsilon,
    )


def _coerce_atom_species(
    atom_species: np.ndarray | list[int] | list[str] | None,
    n_atoms: int,
) -> np.ndarray:
    if atom_species is None:
        return np.zeros(n_atoms, dtype=int)
    arr = np.asarray(atom_species)
    if len(arr) != n_atoms:
        raise ValueError("atom_species must have one entry per atom")
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int)
    species_to_idx = {symbol: idx for idx, symbol in enumerate(sorted(set(arr.tolist())))}
    return np.array([species_to_idx[symbol] for symbol in arr.tolist()], dtype=int)
