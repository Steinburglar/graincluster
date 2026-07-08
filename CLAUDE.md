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

## Objective (Updated: Bayesian Marginal Likelihood)

```
L = (1 − β) · Σ_C L_data(C) + β · Σ_cut s_ij + γ · K
```

where `L_data(C)` is the Bayesian marginal code length:

```
L_data(C) = log Γ(N + αM) − log Γ(αM) − Σ_i [log Γ(n_i + α) − log Γ(α)]
```

- `L_data(C)`: exact MDL "mixture code" length for cluster C's edges under Dirichlet(α) prior
- `N`: total internal edges, `M`: total (pair_type, bin) categories, `α`: Dirichlet concentration
- `n_i`: count of edges in category i; automatically accounts for zero-count bins
- `K`: number of clusters (model-complexity penalty)
- `s_ij = d^2 / (2σ²)`: cut cost for boundary edge (i,j)
- `β ∈ [0,1]`: entropy/cut balance weight

**Differences from prior N·H formulation:**
- L_data is the marginal likelihood (integrating out θ), not a plug-in entropy estimate
- Avoids double-counting: doesn't estimate θ from data then pretend θ is known
- Automatically encodes Occam's razor: includes KL divergence cost of learning distribution from data
- Converges correctly for large N (BIC-like term, not linear in N for concentrated clusters)

## Binning Schemes (Linear and Quantile)

**Linear binning** (default, Freedman-Diaconis):
- `fit_bin_scheme(pair_values)` — raw-space linear bins
- Fixed bin width per pair type, range clipped to [p1, p99]
- Prior: Dirichlet(α,...,α) — uniform over bins
- Physical scale preserved across trajectories

**Quantile binning** (CDF-based):
- `fit_bin_scheme_quantile(pair_values, n_bins=50)` — equal-population bins
- Each bin contains ~1/n_bins fraction of reference data
- Bins frozen at reference quantiles; never updated during clustering
- Prior: currently Dirichlet(α,...,α) — uniform over quantiles
  - **TODO**: implement weighted prior Dirichlet(α·p_global) to encode expected distribution
- Entropy becomes relative to reference distribution structure

Per-pair cutoffs (`--pair-cutoffs C-C=2.0`) are applied before binning; edges beyond the pair-specific cutoff are excluded.

## Scientific Invariants

- **Species-pair identity is part of the information model** — joint entropy
  over (pair_type_idx, bin_idx), not pooled
- **Cluster-level normalization** over total internal edge mass, not per pair
  type — keeps species mixing in the cost
- **Boundary penalty separate from entropy term**
- **Bayesian marginal likelihood** (not plug-in N·H) — provides correct MDL code length
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
optimization. `fit_bin_scheme(pair_values)` or `fit_bin_scheme_quantile(pair_values, n_bins=50)`
are the main entry points. See "Binning Schemes" section above for details.

## The "other" Cluster (cluster_id = -1)

A special permanent background cluster that is:
- Always present (never deleted)
- Excluded from the cluster count K in `objective()` (no gamma penalty)
- Always a valid move target in `greedy_optimize` and `louvain_optimize`
- Initialized with `--init merged` or as a fallback for atoms that don't fit elsewhere

Implements the concept of "unidentified structure/lack of structure" — atoms that don't cohere into well-defined clusters. Particularly useful when singleton init would be blocked by merge barrier (large M) or when a mixed-phase system has atoms that don't naturally cluster.

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

### Sigma-5 [001] twist bicrystal (tests entropy-driven grain separation)

```
/n/holylabs/kozinsky_lab/Users/lsteinberger/systems/cu_bicrystal/
  data/raw/bicrystal_cu_sigma5_twist.extxyz          6400 atoms (relaxed)
  data/raw/bicrystal_cu_sigma5_twist_unrelaxed.extxyz
  data/raw/generate_sigma5_twist.py
analysis/graincluster/run_graincluster.py
```

CSL construction:
- θ = 36.87° (cos=4/5, sin=3/5), CSL vectors E1=a[1,2,0], E2=a[-2,1,0]
- Grain A: P_a=[[4,8,0],[-8,4,0],[0,0,10]] on standard FCC conventional cell
- Grain B: P_b=[[8,4,0],[-4,8,0],[0,0,10]] on ROTATED FCC conventional cell
  (cell_b = cell_a @ R.T, pos_b = pos_a @ R.T)
- DO NOT build grain B by "rotate grain_a positions + reduce mod cell" — this
  always creates exact or near-duplicate atom positions because R^{-1}(V) is a
  grain_a lattice vector for any CSL cell vector V. Build from the rotated cell.
- 3200 atoms/grain, 32.33 × 32.33 Å interface, 36.15 Å grain depth

Run:
```bash
python run_graincluster.py \
    --input ../../data/raw/bicrystal_cu_sigma5_twist.extxyz \
    --init grain_tag --alpha 0.1 --cutoff 3.3
```

Expected (alpha=0.1, cutoff=3.3, beta=0.5, gamma=0.5):
- Rank 0: ~2400 atoms, 100% grain_0, entropy ≈ 0.02 nats (bulk FCC crystal)
- Rank 1: ~2368 atoms, 100% grain_1, entropy ≈ 0.21 nats (bulk FCC crystal)
- Ranks 2–5: 192–416 atoms, interface sublayers, entropy 1.7–2.4 nats
- Ranks 6–133: size-2 clusters at entropy ≈ 5.2 nats (dislocation core pairs)
- Singletons: isolated dislocation core atoms with N=0 internal edges
- Total K ≈ 358, boundary atoms ≈ 2368

Unlike the Sigma-7 or crystal-liquid systems, the two bulk grains ARE separated
because dislocation core atoms at the interface have distinct bond distributions
(compressed/stretched bonds) that form a high-entropy "wall" between the grains.
EMT+FIRE relaxation (42 steps) resolves interface overlaps; use grain_tag init.

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

### LPSC solid electrolyte surface slab

```
/n/holylabs/kozinsky_lab/Users/lsteinberger/systems/battery/
  data/labeled/dec23_complete_dft/dataset.extxyz   7839 frames
  analysis/graincluster/run_graincluster.py
```

Li₆PS₅Cl argyrodite solid electrolyte (species Li/P/S/Cl). The 228-atom frames
are surface slabs: S/P/Cl framework occupies z ∈ [5, 40] Å; Li ions sit on both
surfaces (z<5 and z>40 Å) and fill the interior.

Run:
```bash
python run_graincluster.py --frame-index 0 --no-pbc-z
```

Expected (frame 0, alpha=0.1, cutoff=4.0, sigma=1.0):
- Rank 0: ~138 atoms, S-dominant — bulk LPSC framework + interior Li
- Rank 1: ~47 atoms, Li 98% — surface Li layer (one face)
- Rank 2: ~42 atoms, Li 100% — surface Li layer (other face)
- Total K ≈ 4

`--no-pbc-z` required: z-periodic images bridge both surfaces and collapse K=1
(which IS the correct MDL answer for a fully-periodic single-phase crystal).
The surface Li detection is template-free — PTM cannot label Li-coordination
environments.

Key insight: 5.0 Å cutoff + full PBC → K=1 (all bonds connect to one component).
4.0 Å cutoff + no-pbc-z → K=4 (surface Li layers statistically distinct from bulk).
Singleton merge barrier (H_1edge ≈ 4.4 nats at M≈85) just overcomes long Li-Li bonds
at 3.0–4.0 Å but NOT the shorter first-shell bonds → singletons near surface stay separate.

## Recent Development (Jul 2026)

### Bayesian Marginal Likelihood Replacement
Replaced the plug-in N·H entropy estimator with the exact Bayesian marginal likelihood:

```
L_data(C) = log Γ(N + αM) − log Γ(αM) − Σ_i [log Γ(n_i + α) − log Γ(α)]
```

This is derived from integrating out the multinomial parameter θ under a Dirichlet(α) prior, giving the exact "mixture code" MDL length. The plug-in estimator N·H was an approximation that double-counted data (using data to estimate θ̂, then evaluating code length under θ̂). Key differences:
- Correctly decomposes as E[−log P(n|θ)] + KL(posterior||prior)
- Avoids overfitting penalty built-in via KL term
- For concentrated clusters (small H), grows as log N (not linear)
- Tests pass: 103/103

Implementation in `entropy.py`:
- `data_term()` and `data_term_from_counts()` use lgamma formulation
- `self_information()` unchanged — still uses frozen posterior predictive (correct)
- `cluster_entropy()` retained for reporting only (not used in objective)

### Per-Pair Cutoffs
Added `--pair-cutoffs` CLI option to exclude specific pair-type bonds beyond their cutoff:
```bash
python run_graincluster.py --pair-cutoffs C-C=2.0 C-Si=3.0
```

Useful for materials with distinct bonding regimes (e.g., graphene at 1.42 Å vs graphite second shell at 2.46 Å). Implemented in:
- `builder.py`: per-pair filter in `build_edges()`
- `run_graincluster.py`: command-line parsing and output stem tagging

### Quantile Binning (CDF-based)
Added `--bin-scheme quantile` and `--n-quantile-bins N` options.

**Current state:** Quantile bins are frozen at percentile boundaries; prior is still uniform Dirichlet(α,...,α). This makes entropy "relative to the reference distribution shape" but does NOT encode prior belief that clusters should match the reference.

**TODO:** Implement weighted Dirichlet prior Dirichlet(α·p_global) where p_global are empirical bin frequencies from the reference. This would actually encode "I expect clusters to look like the bulk."

### SiC Melting Simulation Results

**System:** 8000 atoms (4000 C, 4000 Si), frame 0 of a melting trajectory. Graphene column dissociating into Si-rich liquid.

**Best result (linear binning + Bayesian L_Bayes + per-pair cutoffs):**
- Configuration: `--init species --alpha 0.1 --beta 0.75 --gamma 0.0 --pair-cutoffs C-C=2.0`
- K = 375 clusters (338 real + 1 other)
- Rank 0: 5965 atoms, 1973 C + 3992 Si (bulk liquid SiC)
- Rank 1: 995 C (majority of graphene bulk)
- Remaining: 337 small clusters, mostly C at interfaces
- Convergence: 20 rounds

**Quantile binning result (50 bins per pair type):**
- Same params as above but `--bin-scheme quantile`
- K = 842 clusters (841 real + 1 other)
- More fragmentation; uniform prior over quantiles does not penalize deviations from reference
- Indicates that weighted prior is needed to match original intent

**Open questions:**
1. Graphene body (rank 1) remains fragmented into ~1275 fragments when using `--init merged` alone (no species pre-grouping), even with Bayesian L_Bayes. Root cause: topological isolation — interface atoms drain to "other"/"liquid", severing C-C cut-edge paths between graphene fragments.
2. Per-pair cutoff C-C=2.0 essential for graphene detection (excludes second/third coordination shells at 2.46, 2.84 Å). Without it, graphene spreads across too many bins.
3. Species init (`--init species`) provides a good starting point but C and Si still mix in the liquid; no automatic separation without entropy difference.

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
