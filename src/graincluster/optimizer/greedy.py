"""Greedy local-move optimizer.

Each pass over atoms: for every atom, find the neighboring cluster that most
reduces the objective (frozen-model score). Accept if delta < 0.

Also considers moving atom to a new singleton cluster (split move) if that
reduces the objective more than any merge.

Stops when a full pass produces no accepted moves.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.cluster import ClusterState
from ..model.partition import Partition, OTHER_ID


@dataclass
class OptimizeResult:
    n_passes: int
    n_moves: int
    objective_initial: float
    objective_final: float


def _split_if_disconnected(partition: Partition, cluster_id: int) -> int:
    """Split cluster into connected components if an atom move disconnected it.

    BFS over internal edges only. Smaller components become new clusters;
    the largest component keeps the original cluster ID.

    Returns number of new clusters created (0 = already connected).
    """
    c = partition.clusters.get(cluster_id)
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
            for eidx in partition._adj[atom]:
                e = partition.edges[eidx]
                nbr = e.j if e.i == atom else e.i
                if int(partition.atom_labels[nbr]) == cluster_id and nbr not in comp:
                    stack.append(nbr)
        components.append(comp)
        remaining -= comp

    if len(components) == 1:
        return 0

    largest = max(components, key=len)

    n_new = 0
    for small_comp in components:
        if small_comp is largest:
            continue

        new_cid = partition.new_cluster_id()
        new_c = ClusterState(cluster_id=new_cid)
        partition.clusters[new_cid] = new_c
        new_c.atom_ids = small_comp.copy()

        for atom in small_comp:
            partition.atom_labels[atom] = new_cid
            c.atom_ids.discard(atom)
            species_idx = int(partition.atom_species_idx[atom])
            c.remove_atom_species(species_idx)
            new_c.add_atom_species(species_idx)

        # Scan edges from this component; process each edge index exactly once.
        seen_eidx: set[int] = set()
        for atom in small_comp:
            for eidx in partition._adj[atom]:
                if eidx in seen_eidx:
                    continue
                seen_eidx.add(eidx)
                e = partition.edges[eidx]
                nbr = e.j if e.i == atom else e.i
                nbr_label = int(partition.atom_labels[nbr])
                key = (e.pair_type_idx, e.bin_idx)

                if nbr in small_comp:
                    # Internal to new cluster — move count from original.
                    new_c.counts[key] = new_c.counts.get(key, 0) + 1
                    new_c.N += 1
                    cnt = c.counts.get(key, 0) - 1
                    if cnt <= 0:
                        c.counts.pop(key, None)
                    else:
                        c.counts[key] = cnt
                    c.N -= 1
                elif nbr_label == cluster_id:
                    # Cross-component edge (between small_comp and largest).
                    # Was internal to original, now cut — remove from original.
                    cnt = c.counts.get(key, 0) - 1
                    if cnt <= 0:
                        c.counts.pop(key, None)
                    else:
                        c.counts[key] = cnt
                    c.N -= 1

        c.invalidate_entropy()
        new_c.invalidate_entropy()
        n_new += 1

    c.invalidate_entropy()
    return n_new


def greedy_optimize(
    partition: Partition,
    max_passes: int = 100,
    allow_splits: bool = True,
    tol: float = -1e-10,
    exact_below_N: int = 10,
) -> OptimizeResult:
    """Run greedy local-move optimization on partition (in-place).

    Parameters
    ----------
    partition:
        Mutable Partition to optimize. Modified in-place.
    max_passes:
        Maximum number of full atom sweeps.
    allow_splits:
        If True, also consider moving each atom to a new singleton cluster.
    tol:
        Accept move only if delta < tol (strict improvement threshold).
    exact_below_N:
        Use exact (non-frozen) data-term delta when src.N or tgt.N is below
        this threshold.  Corrects frozen-model underestimate for small clusters.
        Default 10.  Set to 0 to disable (pure frozen model).
    """
    obj_initial = partition.objective()
    total_moves = 0

    for pass_idx in range(max_passes):
        moves_this_pass = 0
        n_atoms = len(partition.atom_labels)

        for atom in range(n_atoms):
            best_delta = tol
            best_target = None

            # Collect candidate target clusters from neighboring atoms.
            neighbor_clusters: set[int] = set()
            for eidx in partition._adj[atom]:
                e = partition.edges[eidx]
                nbr = e.j if e.i == atom else e.i
                nbr_cid = int(partition.atom_labels[nbr])
                neighbor_clusters.add(nbr_cid)
            # Exclude current cluster.
            src_id = int(partition.atom_labels[atom])
            neighbor_clusters.discard(src_id)
            # OTHER_ID is always a candidate target (except when already there).
            if src_id != OTHER_ID:
                neighbor_clusters.add(OTHER_ID)

            for cid in neighbor_clusters:
                delta = partition.score_move(atom, cid, exact_below_N=exact_below_N)
                if delta < best_delta:
                    best_delta = delta
                    best_target = cid

            # Split move: move atom to a new singleton cluster.
            if allow_splits:
                new_cid = partition.new_cluster_id()
                delta_split = partition.score_move(atom, new_cid, exact_below_N=exact_below_N)
                if delta_split < best_delta:
                    best_delta = delta_split
                    best_target = new_cid

            if best_target is not None:
                partition.apply_move(atom, best_target)
                moves_this_pass += 1
                # Enforce connectivity: split src if disconnected.
                # OTHER_ID is allowed to be disconnected (background class).
                if src_id != OTHER_ID and src_id in partition.clusters:
                    _split_if_disconnected(partition, src_id)

        total_moves += moves_this_pass
        if moves_this_pass == 0:
            break

    obj_final = partition.objective()
    return OptimizeResult(
        n_passes=pass_idx + 1,
        n_moves=total_moves,
        objective_initial=obj_initial,
        objective_final=obj_final,
    )
