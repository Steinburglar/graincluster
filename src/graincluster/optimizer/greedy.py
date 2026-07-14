"""Greedy local-move optimizer.

Each pass over atoms: for every atom, find the neighboring cluster that most
reduces the objective. Accept if delta < 0.

Also considers moving atom to a new singleton cluster if that reduces the
objective more than any neighbor move.

Stops when a full pass produces no accepted moves.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.partition import Partition, OTHER_ID
from .profiling import LiveProfiler


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
    profiler: LiveProfiler | None = None,
    profile_live: bool = False,
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
    """
    obj_initial = partition.objective()
    total_moves = 0

    for pass_idx in range(max_passes):
        tctx = profiler.time_block("greedy_pass") if profiler is not None else None
        if tctx is not None:
            tctx.__enter__()
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
                delta = partition.score_move(atom, cid)
                if delta < best_delta:
                    best_delta = delta
                    best_target = cid

            # Split move: move atom to a new singleton cluster.
            if allow_splits:
                new_cid = partition.new_cluster_id()
                delta_split = partition.score_move(atom, new_cid)
                if delta_split < best_delta:
                    best_delta = delta_split
                    best_target = new_cid

            if best_target is not None:
                partition.apply_move(atom, best_target)
                moves_this_pass += 1
                if profiler is not None:
                    profiler.add_count("accepted_moves", 1)

        if tctx is not None:
            tctx.__exit__(None, None, None)
        total_moves += moves_this_pass
        if profile_live and profiler is not None:
            for line in profiler.format_checkpoint(
                f"[profile] greedy pass {pass_idx + 1}: moves={moves_this_pass}"
            ):
                print(line, flush=True)
        if moves_this_pass == 0:
            break

    obj_final = partition.objective()
    return OptimizeResult(
        n_passes=pass_idx + 1,
        n_moves=total_moves,
        objective_initial=obj_initial,
        objective_final=obj_final,
    )
