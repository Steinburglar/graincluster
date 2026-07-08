"""Louvain-style optimizer: alternates atom-level greedy sweeps with
cluster-level merge sweeps.

The cluster-merge phase is the aggregation step from Louvain / Leiden: after
atom-level moves converge, every adjacent cluster pair is scored for a full
merge using the exact ΔL formula.  This escapes local minima that greedy
atom moves cannot reach (e.g. two large identical clusters that share no
profitable single-atom move but are cheaper merged).

The outer loop repeats until neither phase makes progress.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.partition import Partition, OTHER_ID
from .greedy import greedy_optimize, OptimizeResult


@dataclass
class LouvainResult:
    n_rounds: int
    n_atom_moves: int
    n_cluster_merges: int
    objective_initial: float
    objective_final: float


def cluster_merge_sweep(
    partition: Partition,
    tol: float = -1e-10,
) -> int:
    """One pass: score every adjacent cluster pair; merge if ΔL < tol.

    Returns number of merges accepted.  Pairs are collected before the sweep
    starts; a check on cluster existence handles cascading merges within one
    pass (earlier merge may remove a cluster referenced by a later pair).
    """
    seen: set[tuple[int, int]] = set()
    pairs: list[tuple[int, int]] = []

    for e in partition.edges:
        ci = int(partition.atom_labels[e.i])
        cj = int(partition.atom_labels[e.j])
        if ci != cj:
            key = (min(ci, cj), max(ci, cj))
            if key not in seen:
                seen.add(key)
                pairs.append(key)

    n_merged = 0
    for cid_a, cid_b in pairs:
        if cid_a not in partition.clusters or cid_b not in partition.clusters:
            continue
        # Never merge OTHER_ID with a real cluster.
        if cid_a == OTHER_ID or cid_b == OTHER_ID:
            continue
        delta = partition.score_cluster_merge(cid_a, cid_b)
        if delta < tol:
            # Absorb the smaller cluster into the larger for efficiency.
            ca = partition.clusters[cid_a]
            cb = partition.clusters[cid_b]
            if len(ca.atom_ids) <= len(cb.atom_ids):
                partition.apply_cluster_merge(src_cid=cid_a, tgt_cid=cid_b)
            else:
                partition.apply_cluster_merge(src_cid=cid_b, tgt_cid=cid_a)
            n_merged += 1

    return n_merged


def louvain_optimize(
    partition: Partition,
    max_rounds: int = 20,
    max_atom_passes: int = 100,
    allow_splits: bool = True,
    exact_below_N: int = 10,
    tol: float = -1e-10,
) -> LouvainResult:
    """Louvain-style optimization (in-place).

    Each round:
      1. Atom-level greedy sweep until convergence.
      2. Cluster-merge sweep (one pass over adjacent pairs).

    Stops when both phases produce no moves in the same round.

    Parameters
    ----------
    partition:
        Mutable Partition to optimize.
    max_rounds:
        Maximum number of (atom sweep + cluster merge) rounds.
    max_atom_passes:
        Max passes per atom-level greedy sweep.
    allow_splits:
        Passed to greedy_optimize.
    exact_below_N:
        Exact delta for small clusters; passed to greedy_optimize.
    tol:
        Accept threshold for both phases.
    """
    obj_initial = partition.objective()
    total_atom_moves = 0
    total_cluster_merges = 0

    for round_idx in range(max_rounds):
        atom_result = greedy_optimize(
            partition,
            max_passes=max_atom_passes,
            allow_splits=allow_splits,
            exact_below_N=exact_below_N,
            tol=tol,
        )
        total_atom_moves += atom_result.n_moves

        n_merges = cluster_merge_sweep(partition, tol=tol)
        total_cluster_merges += n_merges

        if atom_result.n_moves == 0 and n_merges == 0:
            break

    return LouvainResult(
        n_rounds=round_idx + 1,
        n_atom_moves=total_atom_moves,
        n_cluster_merges=total_cluster_merges,
        objective_initial=obj_initial,
        objective_final=partition.objective(),
    )
