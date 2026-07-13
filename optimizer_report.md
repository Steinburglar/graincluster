# Optimizer Report

## Scope

This report describes the current optimization scheme in `graincluster` as
implemented in:

- `src/graincluster/optimizer/greedy.py`
- `src/graincluster/optimizer/louvain.py`
- `src/graincluster/model/partition.py`
- `src/graincluster/model/parameterized.py`
- `src/graincluster/model/cluster.py`

It covers:

- the current objective and state representation
- how atom moves and cluster merges are proposed and scored
- how connectivity is enforced
- current inefficiencies, semantic bugs, and likely improvement paths

## Current Optimization Stack

The optimizer is two-level:

1. atom-level greedy moves
2. cluster-level full merges

This is implemented as a Louvain-style outer loop:

1. run greedy atom sweeps until no atom move is accepted
2. run one merge sweep over adjacent real-cluster pairs
3. repeat until neither phase changes the partition

Files:

- `greedy_optimize(...)` in `src/graincluster/optimizer/greedy.py`
- `cluster_merge_sweep(...)` and `louvain_optimize(...)` in `src/graincluster/optimizer/louvain.py`

## State Representation

`Partition` stores:

- `atom_labels[i]`: cluster id for atom `i`
- `clusters[cid]`: `ClusterState`
- `edges`: graph edges with
  - `pair_type_idx`
  - `bin_idx`
  - `cut_cost`
- `_adj[atom]`: incident edge indices
- `_cut_cost_total`: cached total weighted cut over the current partition

`ClusterState` stores:

- `atom_ids`
- `species_counts[species_idx]`
- `counts[(pair_type_idx, bin_idx)]`
- `N`: total internal edge count

The permanent background cluster is `OTHER_ID = -1`.

Important semantics:

- real clusters are meant to remain connected
- `OTHER_ID` is allowed to be disconnected
- `OTHER_ID` is excluded from the structural cluster count `K`

## Objective

There are two objective paths.

### Legacy Path

Used only when no new structural prior knobs are set:

```text
L_legacy =
    (1 - beta) * sum_C L_data(C)
  + beta * x_cut
  + gamma * K
```

This path is legacy compatibility only.

### Current New Path

Used whenever either structural prior is active:

```text
L =
    sum_C L_data(C)
  + L_K
  + L_cut
```

where:

```text
L_K   = -log P(K)
L_cut = -log p(x_cut | N, K)
```

`L_data(C)` is the cluster identity code:

```text
L_data(C) =
    L_species(C)
  + sum_t L_edge_bins(C, t)
```

Each multinomial block is currently scored as:

```text
L(counts, theta_hat) =
    -log P(counts | theta_hat)
  + [-log p(theta_hat) + (d/2) log N]
```

with:

- exact multinomial coefficient
- exact Dirichlet normalizer
- constrained-MAP or posterior-mean `theta_hat`

## Structural Priors

### Cluster Count

```text
K | lambda ~ Poisson(lambda)
lambda ~ Gamma(a, b)
```

with:

```text
E[lambda] = mu_K
strength  = s_K
```

and current default strength reparameterization:

```text
s_K = N_atoms * 10^(tau_K)
```

`lambda` is marginalized out. There is no asymptotic parameter correction here.

### Weighted Cut

```text
x_cut | lambda, N, K ~ Exponential(lambda)
lambda | N, K ~ Exponential(beta_NK)
beta_NK = beta0 * N^(2/3) * K^(1/3)
```

This is the fixed-shape-1 Lomax prior predictive:

```text
L_cut = log(beta_NK) + 2 log(1 + x_cut / beta_NK)
```

Again, `lambda` is marginalized out. No asymptotic parameter correction belongs
here.

As a strict code length, this term implies a fixed but unspecified cut
resolution `Delta x`; that additive constant is dropped from optimization.

## Atom-Level Optimization

`greedy_optimize(...)` loops over atoms in index order.

For each atom:

1. collect neighboring cluster ids from incident edges
2. remove the current source cluster
3. add `OTHER_ID` as a candidate unless the atom is already in `OTHER_ID`
4. optionally evaluate a fresh singleton cluster id
5. score every candidate with `partition.score_move(...)`
6. accept the best move if `delta < tol`
7. apply the move
8. if the source real cluster disconnected, split it into connected components

The algorithm stops when a full pass accepts no atom move.

### Candidate Set

The move set is local:

- neighboring clusters
- `OTHER_ID`
- one brand-new singleton

It does not consider direct moves into arbitrary non-neighbor real clusters.

This imposes a graph-topological search bias.

## Atom Move Delta

`Partition.score_move(atom, target_cluster_id, ...)` computes:

```text
delta = delta_data + delta_structure
```

for the new prior path, or:

```text
delta = (1 - beta) * delta_data + beta * delta_cut + gamma * delta_K
```

for the legacy path.

### `delta_data`

Computed exactly.

Two cases:

1. no source split after removing the atom
2. source split after removing the atom

#### No-Split Path

Implemented in `_exact_data_delta(...)`.

Procedure:

1. copy only the affected source/target species counts
2. scan incident edges of the moved atom
3. identify touched pair types
4. update only the touched source/target per-pair-type bin counts
5. recompute only the changed multinomial blocks:
   - source species
   - target species
   - touched source edge-bin blocks
   - touched target edge-bin blocks
6. return `after_changed_blocks - before_changed_blocks`

This path is exact and avoids recomputing untouched pair-type blocks.

#### Split-Aware Path

Implemented in `_exact_data_delta_with_split(...)`.

If removing the atom disconnects the source real cluster:

1. build the connected components of the source after removal
2. build exact species counts and exact internal edge-bin counts for each
   resulting component
3. build the exact target-after counts
4. compute:
   - `L_data(src_before) + L_data(tgt_before)`
   - `sum L_data(source_components_after) + L_data(tgt_after)`
5. return the difference

This path is exact for the connected-partition semantics enforced by the
optimizer.

### `delta_K`

Computed from direct cluster creation/deletion only:

- `+1` if target is a new real cluster
- `-1` if source becomes empty and source is not `OTHER_ID`

`OTHER_ID` never contributes to `K`.

### `delta_cut_raw`

Computed in `_cut_cost_delta_for_move(...)` by scanning incident edges:

- source-internal edges become cut: `+cut_cost`
- target-cut edges become internal: `-cut_cost`

This is exact for the direct move itself. If the source splits, an additional
term is added for edges that become cut between the new source components.

### `delta_structure`

New prior path:

```text
delta_structure =
    [L_K(after) - L_K(before)]
  + [L_cut(after) - L_cut(before)]
```

using:

- `K_after = K_before + delta_K`
- `x_cut_after = x_cut_before + delta_cut_raw`

## Connectivity Repair

After an accepted move, `Partition.apply_move(...)` repairs source connectivity
internally unless the source was `OTHER_ID`.

This function:

1. finds connected components of the source cluster by BFS over internal edges
2. keeps the largest component under the original cluster id
3. turns each smaller component into a new cluster
4. updates:
   - `atom_labels`
   - `atom_ids`
   - `species_counts`
   - internal edge counts
   - cached total cut `_cut_cost_total`

This repair is now part of the partition-level move primitive, and the scored
move delta anticipates its effect exactly when a source split would occur.

## Cluster-Level Merge Sweep

After atom moves converge, `cluster_merge_sweep(...)`:

1. collects all adjacent cluster pairs from cut edges
2. includes `OTHER_ID` adjacency when scoring merges
3. scores each pair with `score_cluster_merge(...)`
4. merges if `delta < tol`
5. absorbs:
   - a real cluster into `OTHER_ID`, when one side is `OTHER_ID`
   - otherwise the smaller cluster into the larger one

This is one merge pass per Louvain round.

## Merge Delta

`score_cluster_merge(cid_a, cid_b)` computes:

1. choose the smaller cluster to scan
2. find all cross edges between the two clusters
3. accumulate:
   - `between_counts[(pair_type, bin)]`
   - `between_cut_cost`
4. form merged edge counts:
   - internal counts of `A`
   - plus internal counts of `B`
   - plus the former cross edges
5. form merged species counts:
   - species counts of `A`
   - plus species counts of `B`
6. compute:
   - `L_data(merged)`
   - `L_data(A) + L_data(B)`
7. add structural delta:
   - `K -> K - 1`
   - `x_cut -> x_cut - between_cut_cost`

So under the new prior path:

```text
delta_merge =
    L_data(merged) - L_data(A) - L_data(B)
  + [L_K(K-1) - L_K(K)]
  + [L_cut(x_cut - between_cut_cost, K-1) - L_cut(x_cut, K)]
```

## High-Severity Findings

At this point, the most serious earlier correctness issues are fixed:

- move scoring now includes source-splitting connectivity semantics
- `_cut_cost_total` is updated during connectivity repair
- `exact_below_N` has been removed from the active optimizer API

The remaining issues are now mostly performance and search-design issues rather
than outright scoring bugs.

## Performance Problems

### 1. Split-Aware Move Scoring Can Still Be Expensive

When a source split is possible, the exact path rebuilds connected components
and exact component count tables. This is correct, but it can be expensive for
large source clusters.

### 2. Exact Recompute Per Candidate Move

Every candidate move:

- rescans incident edges
- rebuilds changed local count structures
- evaluates exact multinomial block costs

The no-split path is now tighter than before because it only recomputes touched
species and pair-type blocks, not every block in the source and target
clusters.

This is expensive, especially with many pair types and bins.

### 3. Repeated Pair-Type Filtering

`cluster_data_term(...)` calls `edge_counts_for_pair_type(...)`, which filters
the entire sparse edge-count dictionary separately for every pair type.

This adds an avoidable `O(T * nnz_cluster)` factor.

### 4. `_data_term_from_parts(...)` Rebuilds Dense Arrays Repeatedly

For every pair type:

- allocate a fresh dense array
- iterate all sparse edge counts
- pick out only matching entries

This is another avoidable repeated sparse-to-dense conversion cost.

### 5. Connectivity BFS After Every Accepted Move

`_split_cluster_if_disconnected(...)` runs a BFS over the whole source cluster after
every accepted move.

For large clusters, this can dominate runtime.

### 6. Singleton Candidate Burns Cluster IDs

`greedy_optimize(...)` calls `new_cluster_id()` just to score a hypothetical
singleton. Unaccepted proposals still consume ids.

This is not a correctness issue, but it is sloppy state evolution.

## Structural Search Limitations

### 1. Only Adjacent Real-Cluster Merges

Merge sweep only considers cluster pairs already adjacent by at least one cut
edge in the current partition.

If two meaningful regions are separated through `OTHER_ID`, there is no direct
merge move.

### 2. Local Candidate Set

Atom moves are only into:

- neighboring clusters
- `OTHER_ID`
- a singleton

There is no long-range reassignment move.

## Semantics / Interface Debt

### 1. Legacy `beta` / `gamma` Still Leak Through Interfaces

The new path is additive in proper prior terms, but some helpers still expose
legacy `beta` and `gamma` because of backward compatibility.

This is manageable but should be narrowed over time.

## Recommended Improvement Order

1. Keep the split-aware move regression tests.
   - current tests now cover source-splitting moves and should remain

2. Refactor count storage by pair type.
   - store nested arrays or per-pair-type dense counts
   - stop rebuilding dense vectors from sparse maps on every local score

3. Consider faster split-aware local deltas.
   - current no-split path already uses touched-block exact deltas
   - the split path is still more expensive and could use better local caching

4. Consider stronger merge proposals through `OTHER_ID`.
   - `OTHER_ID` absorption now exists
   - but only for adjacent clusters, not for same-phase regions separated
     through `OTHER_ID`

## Bottom Line

The current optimizer is now:

- exact on local move deltas under the connected-partition semantics enforced by
  the optimizer
- exact on direct cluster-merge deltas
- exact on split-aware cut deltas

The current optimizer is semantically much cleaner than before. The main
remaining issues are:

- expensive exact local rescoring
- local search limitations
- legacy interface scaffolding
