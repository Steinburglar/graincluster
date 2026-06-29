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

**M and the singleton merge barrier** (see section below): M controls how
hard it is to bootstrap clusters from singleton init. Structures with broad
bond distributions (atom overlaps, mixed crystal+liquid) → smaller M → lower
barrier. Pure crystal in a correct bicrystal → large M → barrier not overcome
by bulk FCC bonds with σ=1.0.

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

**Barrier vs cut cost for FCC Cu, σ=1.0:**

For singleton init to bootstrap, need `cut_cost(d) + gamma > H_1edge(M, alpha)`.
With d=2.556 Å (FCC NN), sigma=1.0, gamma=0.5: cut_cost = 3.267 nats.

| M bins | alpha=0.1 H_1edge | Barrier overcome? |
|--------|-------------------|-------------------|
| 44     | 3.50 nats         | barely (delta=-0.27) |
| 50     | 3.73 nats         | no (delta=+0.11) |
| 159    | 5.07 nats         | no (delta=+1.3) |
| 256    | 5.55 nats         | no (delta=+1.8) |

Critical rule: **singleton init only works when M < ~50 for FCC Cu with σ=1.0**.
Broader bond distributions (e.g. unrelaxed bicrystal with atom overlaps) give
few bins → barrier overcome. Correctly constructed bicrystals or pure crystal
have M > 100 → barrier NOT overcome from singleton init.

**Recommended init strategy:**
- `grain_tag` — correct when grain labels are in the input (use for validation)
- `singleton` — use only with sigma much larger than interatomic distances, OR
  with max_bins capped so M < 50
- `merged` — rarely useful; cut costs prevent atoms from escaping

The original 250-atom unrelaxed bicrystal gave M=44 (atom overlaps widened IQR)
→ singleton init worked accidentally. Correct CSL bicrystals have M=159+ →
singleton init fails for bulk crystal atoms.

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

Note: alpha alone cannot fix the singleton barrier for FCC Cu with M > 50. The
critical alpha that barely allows FCC merger is alpha ≈ 0.004 (for M=256), but
this also weakens entropy signal. Prefer `grain_tag` init over tuning alpha.

## Validation Datasets

### Crystal-liquid interface (recommended)

```
/n/holylabs/kozinsky_lab/Users/lsteinberger/systems/cu_bicrystal/
  data/raw/bicrystal_cu_crystal_liquid.extxyz   1280 atoms
  analysis/graincluster/run_graincluster.py
```

FCC Cu crystal (grain tag 0, 640 atoms) + MD-melted liquid grain (grain tag 1,
640 atoms). Generated by `data/raw/generate_crystal_liquid.py`.

Liquid generation: same 8×8×10 FCC Cu slab as crystal, Langevin NVT MD at
1500 K (EMT, well above EMT T_melt ~900 K) for 4 ps. Gives a disordered
liquid snapshot with physical first-shell peak and realistic bond distribution.
Interface geometry: explicit crystal-liquid gap ≈ 2.26 Å (bonds form at
cutoff 3.3 Å), PBC gap = 3.5 Å (no spurious bonds at z-boundary).

Run:
```bash
python run_graincluster.py \
    --input ../../data/raw/bicrystal_cu_crystal_liquid.extxyz \
    --init grain_tag --alpha 0.1 --cutoff 3.3
```

Expected (alpha=0.1, cutoff=3.3, sigma=1.0):
- Rank 0: 641 atoms, 100% grain 0 (crystal), entropy ≈ 1.14 nats
- Liquid: ~10 major clusters (sizes 29–60, entropy ~4.0–4.3 nats) + many
  singletons. This is correct MDL behavior: the liquid has 174 bins (broad
  distribution), and local patches have slightly different bond statistics →
  optimizer prefers splitting over paying the full uniform-distribution entropy.
  Key validation signal: crystal entropy << liquid entropy (1.1 vs 4.0+ nats).
- Total K ≈ 188

Liquid fragmentation is a fundamental MDL property, not a bug. The liquid grain
spans many entropy bins, and local sampling fluctuations give slightly lower
description length for split clusters than for one unified high-entropy cluster.
Singletons with N=0 internal edges are atoms fully surrounded by cluster
boundaries — not isolated atoms with no bonds.

Use `grain_tag` init because bulk FCC bonds (d=2.556 Å) cannot overcome the
singleton merge barrier for M≈174 bins with σ=1.0.

### Sigma-7 CSL twist bicrystal (tests topological behavior, NOT entropy signal)

```
data/raw/bicrystal_cu_sigma7_large_relaxed.extxyz   1120 atoms
analysis/graincluster_large/run_graincluster.py
```

Note: Both grains are FCC Cu with identical bond distributions. The algorithm
correctly treats them as one entropy cluster (they ARE informationally
equivalent — same bond statistics). The grain boundary IS detected as
high-cut-cost edges at the explicit interface gap. This tests the boundary
penalty term, not entropy-driven separation.

### Original small bicrystal (DEPRECATED, do not use for new validation)

```
data/raw/bicrystal_cu_unrelaxed.extxyz   250 atoms
```

Worked with singleton init only because atom overlaps at the interface
gave M=44 bins (IQR inflated by ~0.1 Å phantom bonds), accidentally placing
the barrier below FCC cut cost. Not physically valid.

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
