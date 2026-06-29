# graincluster — Agent Context

## Purpose

MDL-style configuration-space segmentation for atomistic simulations.
Finds contiguous regions (phase pockets, grain boundaries) whose internal
edge statistics are low-information relative to the rest of the system.

This is **not** a local environment labeler. It is a graph partitioning
engine driven by a description-length objective.

## Relationship to graphcluster

`graincluster` imports `graphcluster.io.frame.Frame` as a data carrier.
It does **not** reuse the graphcluster runner, partitioner, or optimizer.
The objective, data model, and optimization backend are all different.
graphcluster uses leidenalg with preset objectives; graincluster implements
its own Louvain-style optimizer against the MDL objective.

## Objective

```
L = sum_C N_C * H_C + gamma * K + lambda_cut * sum_cut s_ij
```

- `N_C`: total internal edge count for cluster C
- `H_C`: joint entropy over (pair_type, bin) with Dirichlet smoothing
- `K`: number of clusters (model-complexity penalty)
- `s_ij = d^2 / (2 * sigma^2)`: cut cost for boundary edge (i,j)

## Scientific Guardrails (do not change these)

- **Raw-space linear bins**, not CDF/percentile bins — physical scale must
  be preserved across the trajectory
- **Species-pair identity is part of the information model** — joint entropy
  over (pair_type_idx, bin_idx), not pooled
- **Cluster-level normalization** over total internal edge mass, not per pair
  type — keeps species mixing in the cost
- **Boundary penalty separate from entropy term**
- **Dirichlet smoothing** (alpha > 0) — prevents log(0) singularities
- **Frozen empirical model for move scoring** — move deltas use pre-move
  cluster states; counts updated only after acceptance. Exact path used
  for small clusters (N < exact_below_N=10 by default).

## Package Layout

```
src/graincluster/
  graph/
    edge.py         EdgeRecord dataclass (i, j, pair_key, pair_type_idx,
                    raw_value, bin_idx, cut_cost)
    builder.py      build_edges() — cKDTree neighbor search → EdgeRecord list
  features/
    species.py      canonical_pair_key(), all_pair_keys()
    binning.py      PairBinScheme, BinScheme, fit_bin_scheme()
                    Freedman-Diaconis bin count per pair type
  model/
    cluster.py      ClusterState — joint count table n_(t,b), N, atom_ids
    entropy.py      cluster_entropy(), data_term(), data_term_from_counts(),
                    self_information()
    partition.py    Partition — full clustering state; score_move(),
                    apply_move(), score_cluster_merge(), apply_cluster_merge(),
                    objective(), partition_from_labels()
  optimizer/
    greedy.py       greedy_optimize() — atom-level local move sweep
    louvain.py      louvain_optimize(), cluster_merge_sweep() — Louvain-style
                    outer loop: atom sweep + cluster-merge phase
```

## Key Data Structures

**ClusterState** — mutable; owns `counts: dict[(pair_type_idx, bin_idx), int]`
and `N` (total internal edges). Call `add_edge` / `remove_edge` to update;
these invalidate `_entropy`.

**Partition** — owns `atom_labels` (np.ndarray, int), `clusters` dict,
`edges` list, `_adj` (atom → edge indices). `apply_move` and
`apply_cluster_merge` are the only correct ways to mutate a live partition;
they keep `_adj`, counts, and atom_ids consistent.

**BinScheme** — fit once from a reference edge sample; bins are fixed during
optimization. `fit_bin_scheme(pair_values)` takes `dict[pair_key, np.ndarray]`.

## Binning

`fit_bin_scheme` uses Freedman-Diaconis per pair type:
- `h = 2 * IQR * N^(-1/3)`
- bin count `B = ceil(span / h)`, clamped to [min_bins=4, max_bins=256]
- range clipped to [p1, p99] of reference data
- degenerate IQR (`< 1e-6`): falls back to `sqrt(N)` bins

**IQR degeneracy pitfall**: floating-point IQR can be ~1e-8 (not exactly 0)
when >50% of bonds are at identical distances (e.g. perfect FCC crystal).
The check must be `iqr < 1e-6`, not `iqr <= 0.0`, to catch this.

FD is scale-invariant for symmetric unimodal distributions: bin COUNT is the
same for narrow vs. broad distributions; what changes is bin WIDTH. For other
distribution shapes spread does affect bin count.

## Merge Barrier

The cost of adding the first edge to an empty cluster is `H_1edge ≈ log(M)`
nats, where M = total bin categories. This is the "singleton merge barrier":
two singleton clusters merge only when `lambda_cut * cut_cost + gamma > H_1edge`.

- Barrier is O(log M), not O(log N) — calibrated to statistical resolution,
  not sample size.
- Reducing alpha lowers the barrier (weaker Dirichlet prior, H_1edge → 0),
  but too-small alpha also kills entropy signal in large clusters (degeneracy).
- Default alpha=0.5 is a reasonable starting point; alpha=0.01–0.1 lowers the
  barrier and can improve singleton-init convergence.

## Move Scoring

### Atom-level (frozen model)

`score_move(atom, target_cluster_id, exact_below_N=10)` computes ΔL without
mutating state:

1. Classify each incident edge of `atom`:
   - neighbor in src → edge leaves src (becomes cut)
   - neighbor in target → edge enters target (becomes internal)
2. If `src.N < exact_below_N` or `tgt.N < exact_below_N`: use exact path
   (`_exact_data_delta`) — builds post-move count tables in temp dicts and
   recomputes data_term exactly. Corrects frozen-model underestimate for
   last-edge-removal from small clusters.
3. Otherwise frozen model: `delta_entropy = Σ I_tgt(e entering) - Σ I_src(e leaving)`
   where `I_C(t,b) = -log(p̃_{t,b}(C))`
4. `delta_cut = lambda_cut * (cut_costs_gained - cut_costs_lost)`
5. `delta_K = gamma * (+1 if target is new, -1 if src becomes empty)`

### Cluster-level (exact, Louvain aggregation phase)

`score_cluster_merge(cid_a, cid_b)` computes exact ΔL for absorbing one
whole cluster into another:

```
ΔL = data_term(A∪B) - data_term(A) - data_term(B)
     - lambda_cut * Σ(cut_costs between A and B)
     - gamma
```

Scans atoms of the smaller cluster to find cross edges in O(|smaller| * degree).
`apply_cluster_merge(src, tgt)` absorbs src into tgt: cross edges become
internal, src counts transfer to tgt, atom labels updated, src deleted.

## Optimizer: Louvain-style (default)

`louvain_optimize(partition)` in `optimizer/louvain.py` is the recommended
entry point. Each round:

1. **Atom sweep** (`greedy_optimize`): sweep all atoms, score moves to
   neighboring clusters and new singletons, accept if `delta < tol`.
2. **Cluster-merge sweep** (`cluster_merge_sweep`): collect all adjacent
   cluster pairs from cut edges, score each merge exactly, accept if
   `delta < tol`. Smaller cluster absorbed into larger.

Repeats until both phases produce zero moves. This escapes local minima that
atom moves alone cannot reach — e.g. two large identical clusters where any
single atom move increases cut cost, but the full merge reduces it.

`greedy_optimize` alone is available for cases where cluster-merge is not
needed.

## alpha Behavior

| alpha | Effect |
|-------|--------|
| 0.5 (default) | Smooth posteriors, barrier ≈ log(M) nats |
| 0.1 | Lower barrier, faster singleton merge |
| 0.01 | Very low barrier, but near-zero entropy signal for large crystal clusters → Louvain cluster-merge phase needed to fix fragmentation |
| 0 | Degeneracy: zero self-information for dominant bins, optimizer stalls |

With alpha=0.01, greedy alone may fragment a pure crystal into sub-clusters
(all zero-entropy, no merge signal). The cluster-merge phase corrects this
in one sweep because `ΔL = ΔL_data − gamma < 0` for identical-distribution
adjacent clusters.

## Validation Dataset

FCC Cu twist-boundary bicrystal at:
```
/n/holylabs/kozinsky_lab/Users/lsteinberger/systems/cu_bicrystal/
  data/raw/bicrystal_cu_unrelaxed.extxyz   250 atoms, unrelaxed
  analysis/graincluster/run_graincluster.py
```

Run:
```bash
python run_graincluster.py --init singleton --alpha 0.1
```

Expected: K=2, 100% grain purity on both clusters.

ASE GUI coloring: `View → Colors → Tag` (cluster rank, instant) or
`View → Colors → User-defined → cluster_color_code` (hashed, like graphcluster).

## Running Tests

```bash
/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311/bin/python \
    -m pytest tests/ -q
```

Tests are in `tests/`:
- `test_binning.py` — FD rule, PairBinScheme assign, fit roundtrips
- `test_entropy.py` — ClusterState mutations, entropy math, self_information
- `test_moves.py` — score_move correctness (frozen + exact), apply_move state consistency
- `test_optimizer.py` — greedy convergence, monotone objective, phase separation
- `test_integration.py` — end-to-end: crystal < liquid entropy, interface detection,
  FD bin width scaling
- `test_louvain.py` — score_cluster_merge exactness, apply_cluster_merge state
  consistency, cluster_merge_sweep behavior, local-minimum escape

## Implementation Status

| Milestone | Status |
|-----------|--------|
| 0 — repo scaffold | done |
| 1 — core data model | done |
| 2 — feature extraction + FD binning | done |
| 3 — entropy model | done |
| 4 — move scoring (frozen + exact small-cluster path) | done |
| 5 — greedy optimizer | done |
| 5b — Louvain cluster-merge phase | done |
| 6 — validation tests (96 tests) | done |
| 7 — streaming trajectory, plots | not started |

## What Is Not Yet Built

- `io/` module — trajectory reading (reuse graphcluster readers or add thin wrapper)
- `analysis/` module — cluster summaries, per-cluster entropy reports, plots
- Streaming trajectory runner (process multiple frames in sequence)
- CLI entry point

## Coding Conventions

- Edit `partition.py` → always verify `_adj`, `atom_labels`, `clusters`, and
  edge counts remain consistent after any mutation
- New objective terms → add to `Partition.objective()`, `score_move()`, AND
  `score_cluster_merge()` together; update tests in `test_moves.py` and
  `test_louvain.py`
- New bin schemes → must implement `assign(pair_key, values)` and
  `assign_one(pair_key, value)` on `BinScheme`
- `apply_cluster_merge`: scan cross edges BEFORE updating atom labels
  (labels used to identify cross edges; updating first would break detection)
- After any file edit: `git diff -- <file>` to verify
