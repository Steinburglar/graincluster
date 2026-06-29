"""Greedy local-move optimizer.

Each pass over atoms: for every atom, find the neighboring cluster that most
reduces the objective (frozen-model score). Accept if delta < 0.

Also considers moving atom to a new singleton cluster (split move) if that
reduces the objective more than any merge.

Stops when a full pass produces no accepted moves.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.partition import Partition


@dataclass
class OptimizeResult:
    n_passes: int
    n_moves: int
    objective_initial: float
    objective_final: float


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
