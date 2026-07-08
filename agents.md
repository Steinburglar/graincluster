# graincluster — Agent Guide

## What It Does

MDL-style graph partitioning engine for atomistic simulations. Finds contiguous regions (phases, grain boundaries, interfaces) whose internal bond statistics are low-information relative to the rest of the system. Output: atom labels specifying which cluster each atom belongs to.

**Not** a local environment labeler. Not a nearest-neighbor classifier. It's a global graph optimization tool.

## The Objective

```
L = (1 − β) · Σ_C L_data(C) + β · Σ_cut s_ij + γ · K
```

- `L_data(C)` = Bayesian marginal likelihood (lgamma formulation, see below)
- `β` = entropy/cut balance (typically 0.5–0.9)
- `γ` = cluster count penalty (typically 0, set via `--gamma`)
- `s_ij` = cut cost per boundary edge (depends on distance and σ)

**L_data(C)** is exact Dirichlet-Multinomial mixture code:
```
L_data(C) = log Γ(N + αM) − log Γ(αM) − Σ_i [log Γ(n_i + α) − log Γ(α)]
```

This is NOT the old N·H entropy. It integrates out the multinomial parameter θ, automatically encoding Occam's razor (cost of learning the distribution from data). For concentrated clusters, grows as log N, not linear in N.

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
- Methods: `add_edge()`, `remove_edge()`

**BinScheme** (`features/binning.py`):
- Fixed per-pair-type binning, frozen at initialization
- Two options: linear (Freedman-Diaconis) or quantile (CDF-based)
- Prior: currently Dirichlet(α,...,α) uniform over bins

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
Equal-population bins, frozen at percentiles. Prior: currently uniform (TODO: weighted).

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
    entropy.py           data_term() — Bayesian marginal likelihood (lgamma)
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
Best result: species init + Bayesian L_Bayes + C-C=2.0 cutoff gives K=375 clusters, with one large graphene cluster (995 C) and many smaller fragments (rank 2+).

With `--init merged`, get K=1275 fragments instead. Root cause: interface C atoms drain to "other"/"liquid", cutting off C-C paths between graphene fragments. Cluster-merge sweep can't reconnect them (no cross-edges to propose merges).

**Not a bug in move scoring.** A topological issue: the objective is doing what it's designed to do — finding disconnected regions.

### Quantile Binning Over-Fragments
`--bin-scheme quantile` gives K=842 vs K=375 for linear. Reason: uniform Dirichlet(α,...,α) over quantiles does not penalize deviations from reference distribution.

**TODO**: Implement weighted prior Dirichlet(α·p_global) where p_global are empirical bin frequencies in the reference. This would encode "clusters should match the bulk distribution" and reduce fragmentation.

### Mixed-Phase Entropy
In SiC, rank 0 (bulk liquid SiC) is 1973 C + 3992 Si. No automatic species separation in the liquid because C and Si have overlapping bond distributions.

Consistent with MDL: mixed phases have high entropy regardless of species. Species identity is in the edge type, not the atom label.

## Recent Changes (Jul 2026)

1. **Bayesian marginal likelihood** — replaced N·H with exact lgamma formulation
2. **Per-pair cutoffs** — `--pair-cutoffs C-C=2.0` support
3. **Quantile binning** — `--bin-scheme quantile` added
4. **OTHER_ID cluster** — permanent background implemented
5. **SiC experimental results** — documented in CLAUDE.md

## Constraints to Respect

- **Partition mutations**: only use `apply_move()` and `apply_cluster_merge()` — they maintain consistency
- **Move scoring**: frozen model — uses pre-move cluster states, counts updated only after acceptance
- **Exact path**: small clusters (N < exact_below_N) use exact entropy computation, not frozen model
- **Bin schemes**: frozen at fit time, never updated during optimization
- **Alpha behavior**: larger α lowers merge barrier but also weakens entropy signal for concentrated clusters

## Next Steps for Agents

1. **Weighted Dirichlet prior** — modify `fit_bin_scheme_quantile()` and entropy computations to use Dirichlet(α·p_global)
2. **Graphene fragmentation** — investigate topological reconnection strategies (post-hoc merging, "other"-bridged merging)
3. **Streaming trajectory** — process multiple frames in sequence
4. **Per-phase priors** — detect phase type and use phase-specific prior (crystal vs liquid)
