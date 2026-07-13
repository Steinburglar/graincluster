# graincluster — Agent Guide

## What It Does

MDL-style graph partitioning engine for atomistic simulations. Finds contiguous regions (phases, grain boundaries, interfaces) whose internal bond statistics are low-information relative to the rest of the system. Output: atom labels specifying which cluster each atom belongs to.

**Not** a local environment labeler. Not a nearest-neighbor classifier. It's a global graph optimization tool.

## The Objective

```
L = (1 − β) · Σ_C L_data(C) + β · Σ_cut s_ij + γ · K
```

- `L_data(C)` = target parameterized MAP data cost, see below
- `β` = entropy/cut balance (typically 0.5–0.9)
- `γ` = cluster count penalty (typically 0, set via `--gamma`)
- `s_ij` = cut cost per boundary edge (depends on distance and σ)

### Current Target Model

Target model is **not** Bayesian marginal likelihood. Marginal likelihood
integrates out all cluster parameters:

```
P(D | M) = Π_C ∫ P(D_C | θ_C) P(θ_C) dθ_C
```

This is valid math but wrong semantics for this project: every cluster has the
same identity under the generative model, and self-consistent phases are only
rewarded if they have high universal prior-predictive probability.

Target model is parameterized MAP:

```
P(D, θ | M) = P(D | θ, M) P(θ | M)
```

Each cluster has explicit identity parameters:

```
θ_species,C  = multinomial over atom species
θ_edge,C,t   = multinomial over edge bins for induced pair type t
```

Cluster data term:

```
L_data(C) =
    -log P(species_C | θ_species,C) - log P(θ_species,C)
  + Σ_t [-log P(edge_bins_C,t | θ_edge,C,t) - log P(θ_edge,C,t)]
```

Pair type is deterministic from endpoint species. Do not treat edge pair type
as an independent edge observation.

### Priors

Use Dirichlet priors:

```
α_i = κ · base_i
```

Species:

```
base_species_s = global atom fraction of species s
α_species_s = κ_species · base_species_s
```

Edge bins:

```
base_edge_t,b = 1 / B_t
α_edge_t,b = κ_edge / B_t
```

There is one edge-bin Dirichlet per pair type, but all pair types share
`κ_edge` initially. Quantile bins make uniform edge-bin base correspond to the
global real-space edge-length distribution conditional on pair type.

Use constrained MAP for `θ` during optimization:

```
θ_hat_i >= ε
sum_i θ_hat_i = 1
```

Positive effective weights `n_i + α_i - 1` share non-floor mass. Nonpositive
weights sit at `ε`. Posterior mean remains available as optional diagnostic.

MAP caveat: high `κ` shrinks `θ_hat` toward the base distribution but does not
guarantee mixed/broad finite data have lower profiled cost than pure/narrow
data. It reduces the sparse-data advantage.

Detailed implementation plan:

- `parameterized_model_plan.md`

### Current Code Caveat

Current implementation still uses old marginalized Dirichlet-multinomial edge
code:

```
L_data(C) = log Γ(N + αM) − log Γ(αM) − Σ_i [log Γ(n_i + α) − log Γ(α)]
```

Treat this as baseline/stale implementation, not target science.

## Key Data Structures

**Partition** (`model/partition.py`):
- `atom_labels[i]` = cluster ID for atom i
- `clusters[cid]` = ClusterState with edge counts and atom IDs
- `edges[]` = full edge list (EdgeRecord objects)
- Methods: `score_move()`, `apply_move()`, `score_cluster_merge()`, `apply_cluster_merge()`, `objective()`
- **Invariant**: `_adj` (atom→edges map), `atom_labels`, and cluster edge counts must remain consistent after mutations

**ClusterState** (`model/cluster.py`):
- `counts[(pair_type_idx, bin_idx)]` = edge count in each category
- `N` = total internal edges
- target model must also track atom-species counts
- Methods: `add_edge()`, `remove_edge()`

**BinScheme** (`features/binning.py`):
- Fixed per-pair-type binning, frozen at initialization
- Two options: linear (Freedman-Diaconis) or quantile (CDF-based)
- Target edge prior: Dirichlet with uniform base over pair-type-specific bins

## Optimizer

`louvain_optimize(partition)` is the main entry point:

1. **Atom sweep** — greedy local moves, accept if delta < 0
2. **Cluster-merge sweep** — exact merge scoring, accept if delta < 0
3. Repeat until convergence

Escapes local minima that atom moves alone cannot reach (e.g., two identical clusters that should merge).

## Binning

Two schemes:

**Linear (Freedman-Diaconis, default)**:
```bash
--bin-scheme linear
```
Raw-space bins with width h = 2·IQR·N^(-1/3). Prior: uniform over bins.

**Quantile (CDF-based)**:
```bash
--bin-scheme quantile --n-quantile-bins 50
```
Equal-population bins, frozen at percentiles. Target prior: uniform base over
quantile bins per pair type, with shared `κ_edge`.

**Per-pair cutoffs**:
```bash
--pair-cutoffs C-C=2.0 Si-Si=3.0
```
Exclude bonds beyond pair-specific cutoff before binning.

## The "other" Cluster (ID = -1)

Special permanent background cluster:
- Always exists, never deleted
- Excluded from K (no γ penalty)
- Always a valid move target
- Use `--init merged` to start everything in "other", or `--init species` to pre-group atoms

## Code Layout

```
src/graincluster/
  graph/
    builder.py           build_edges() — neighbor search, per-pair cutoff filtering
  features/
    binning.py           fit_bin_scheme(), fit_bin_scheme_quantile()
    species.py           canonical_pair_key()
  model/
    entropy.py           old marginalized baseline data term (lgamma)
    parameterized.py     target parameterized MAP scoring (to add)
    partition.py         Partition class, score_move(), apply_move(), objective()
    cluster.py           ClusterState — mutable edge count tables
  optimizer/
    louvain.py           louvain_optimize() — main optimizer
    greedy.py            greedy_optimize() — atom sweep only
```

## How to Run

```bash
python /n/holylabs/kozinsky_lab/Users/lsteinberger/systems/sic/analysis/graincluster/run_graincluster.py \
    --frame-index 0 \
    --specorder C Si \
    --cutoff 3.0 \
    --pair-cutoffs C-C=2.0 \
    --init species \
    --gamma 0.0 --alpha 0.1 --beta 0.75 \
    --bin-scheme linear
```

Output: `.traj` and `.extxyz` files with atom cluster assignments.

View in ASE:
```bash
ase gui output.traj
# View → Colors → Tag (cluster ID, discrete)
# View → Colors → Charge (cluster ID, gradient)
```

## Running Tests

```bash
/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311/bin/python \
    -m pytest tests/ -q
```

Current: 103/103 pass.

## Known Issues & Open Problems

### Graphene Fragmentation in SiC
Old baseline result: species init + Bayesian `L_Bayes` + C-C=2.0 cutoff gives K=375 clusters, with one large graphene cluster (995 C) and many smaller fragments (rank 2+).

With `--init merged`, get K=1275 fragments instead. Root cause: interface C atoms drain to "other"/"liquid", cutting off C-C paths between graphene fragments. Cluster-merge sweep can't reconnect them (no cross-edges to propose merges).

**Not a bug in move scoring.** A topological issue: the objective is doing what it's designed to do — finding disconnected regions.

### Quantile Binning Over-Fragments
Old baseline: `--bin-scheme quantile` gives K=842 vs K=375 for linear under the marginalized edge model.

**Old TODO deprecated.** Do not implement joint weighted prior
`Dirichlet(α·p_global)` over `(pair_type, bin)`. New target is atom-species
Dirichlet plus per-pair edge-bin Dirichlet, with explicit cluster parameters.

### Mixed-Phase Entropy
In SiC, rank 0 (bulk liquid SiC) is 1973 C + 3992 Si. No automatic species separation in the liquid because C and Si have overlapping bond distributions.

Old model explanation: species identity was in edge type, not atom label. Target model changes this: species identity is atom-level and charged through species counts.

## Recent Changes (Jul 2026)

1. **Parameterized MAP target selected** — replace marginalized edge model with explicit cluster identities
2. **Per-pair cutoffs** — `--pair-cutoffs C-C=2.0` support
3. **Quantile binning** — `--bin-scheme quantile` added
4. **OTHER_ID cluster** — permanent background implemented
5. **SiC experimental results** — documented in CLAUDE.md

## Constraints to Respect

- **Partition mutations**: only use `apply_move()` and `apply_cluster_merge()` — they maintain consistency
- **Move scoring migration**: implement exact local recompute first for the parameterized model; optimize later
- **Old exact path**: small clusters (N < exact_below_N) use exact old entropy computation, not frozen model
- **Bin schemes**: frozen at fit time, never updated during optimization
- **Kappa behavior**: `κ_species` and `κ_edge` control Dirichlet shape; below category count favors sparse/corner distributions

## Next Steps for Agents

1. **Parameterized MAP model** — implement `parameterized_model_plan.md`
2. **Graphene fragmentation** — investigate topological reconnection strategies (post-hoc merging, "other"-bridged merging)
3. **Streaming trajectory** — process multiple frames in sequence
4. **Per-phase priors** — detect phase type and use phase-specific prior (crystal vs liquid)
