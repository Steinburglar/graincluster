# Parameterized Model Implementation Plan

## Purpose

Replace the current marginalized Dirichlet-multinomial edge model with a
parameterized model with explicit cluster identities:

```text
theta_species,C
theta_edge,C,t
```

The optimizer should choose a partition whose clusters have low data cost under
their own identity parameters, while paying a proper parameter code and a
proper discrete structural prior.

## Target Objective

For partition `M` and cluster parameters `theta`:

```text
L(M, theta) =
    L_partition(M)
  + sum_C L_data(C, theta_C)
```

with:

```text
L_partition(M) = -log P(M)
```

The immediate structural target is:

```text
L_partition(M) = -log P(K(M)) - log p(x_cut(M) | N, K(M))
```

where:

- `K(M)` is the number of real clusters and `P(K)` comes from a
  Poisson-Gamma prior predictive
- `x_cut(M)` is the total weighted boundary cut
- `p(x_cut | N, K)` comes from the fixed-shape-1 Lomax prior predictive
  described below

Cluster data term:

```text
L_data(C, theta_C) =
    L_species(C, theta_species,C)
  + sum_t L_edge_bins(C, t, theta_edge,C,t)
```

Species term:

```text
L_species =
    L(theta_species,C)
  + L(species counts in C | theta_species,C)
```

Edge-bin term for pair type `t`:

```text
L_edge_bins =
    L(theta_edge,C,t)
  + L(edge-bin counts in C for pair type t | theta_edge,C,t)
```

The current implementation now uses an asymptotic two-part parameter code for
`theta`. That is still an approximation, not an exact finite-grid code, but it
is already on the intended path.

## Two-Part Parameter Code

For explicit cluster identities, do not use bare:

```text
-log p(theta_hat)
```

as if it were a codelength. `theta_hat` is continuous, so this is a density,
not a probability mass.

Instead use:

```text
L(theta_hat) + L(D | theta_hat)
```

where `L(theta_hat)` is a finite-resolution code for the chosen parameter.

### Resolution Choice

Arbitrary grid spacing `delta` is not acceptable because finer user-chosen
resolution always increases code length even when the data are unchanged.

For a `d`-dimensional parameter cell:

```text
L(theta_hat) ≈ -log p(theta_hat) - d log delta
```

The intended resolution should therefore be statistical rather than arbitrary.
For regular models with `N` effective observations, parameter uncertainty
scales like:

```text
delta ~ N^(-1/2)
```

leading to an asymptotic MDL/BIC-style code:

```text
L(theta_hat) ≈ -log p(theta_hat) + (d / 2) log N + const
```

This is the appropriate path for:

- `theta_species,C`
- `theta_edge,C,t`

and is the approximation currently used in code.

## Priors

### Structural Prior on Cluster Count

Use a discrete prior on the number of real clusters:

```text
K | lambda ~ Poisson(lambda)
lambda ~ Gamma(a, b)
```

with reparameterization:

```text
E[lambda] = mu_K
strength = s_K
a = s_K
b = s_K / mu_K
```

Treat `lambda` as a nuisance hyperparameter and marginalize it:

```text
P(K) = integral P(K | lambda) P(lambda) d lambda
```

Then:

```text
L_K = -log P(K)
```

is a proper codelength for cluster count.

This is the correct place to use marginalization:

- `theta` stays explicit because cluster identity matters scientifically.
- `lambda` is marginalized because it is only a hyperparameter controlling the
  prior over `K`.

Initial implementation should replace the current `gamma * K` term with this
prior-predictive `L_K`.

### Structural Prior on Weighted Edge Cut

Define the total weighted cut:

```text
x_cut(M) = sum_{(i,j) in boundary(M)} w_ij
```

with fixed Gaussian-kernel edge weights `w_ij`.

Use the fixed-shape "double exponential" hierarchy:

```text
x_cut | lambda, N, K ~ Exponential(lambda)
lambda | N, K ~ Exponential(beta_NK)
```

with:

```text
beta_NK = beta0 * N^(2/3) * K^(1/3)
```

Marginalizing `lambda` gives the shape-1 Lomax prior predictive:

```text
p(x_cut | N, K) = beta_NK / (x_cut + beta_NK)^2
```

Negative log density in the preferred implementation form:

```text
L_cut =
    log(beta_NK)
  + 2 log(1 + x_cut / beta_NK)
```

`cut_prior_beta0 = None` should disable this prior entirely and skip its
computation. A very large `beta0` is the weak-prior limit, but `None` is the
preferred interface for both efficiency and clarity.

Because `x_cut` is continuous data, a literal code length also requires a fixed
resolution `Delta x`:

```text
L_cut_code ≈ -log p(x_cut | N, K) - log Delta x
```

The intended interpretation is that `Delta x` is fixed but unspecified across
candidate partitions, so the additive `-log Delta x` term is constant and is
dropped from optimization. No asymptotic `(d/2) log N` correction belongs
here because `x_cut` is data and the nuisance parameter `lambda` has already
been marginalized out.

This cut prior lives at the same hierarchical level as the `K` prior: it is a
prior factor on one structural summary statistic of the partition, not a
replacement for the cluster identity terms.

### Species Prior

Compute global atom fractions once per frame or reference set:

```text
base_species_s = count_s / N_atoms
alpha_species_s = kappa_species * base_species_s
```

Expose:

```text
kappa_species
```

Expected behavior in two-species 50/50 case:

```text
kappa_species < 2  -> homogeneous clusters favored
kappa_species = 2  -> flat prior over composition
kappa_species > 2  -> theta_hat shrinks strongly toward global composition
```

### Edge Prior

For each pair type `t` and quantile-bin count `B_t`:

```text
base_edge,t,b = 1 / B_t
alpha_edge,t,b = kappa_edge / B_t
```

Expose:

```text
kappa_edge
```

All pair types share `kappa_edge` initially. Each pair type has its own
Dirichlet vector because `B_t` may differ.

Expected behavior:

```text
kappa_edge < B_t  -> narrow/sparse edge-bin distributions favored
kappa_edge = B_t  -> flat prior over edge-bin parameters
kappa_edge > B_t  -> theta_hat shrinks strongly toward global-CDF-like bins
```

## Parameter Estimate

Use floor-constrained MAP for optimization while the code still uses a profiled
energy:

```text
theta_hat_i >= epsilon
sum_i theta_hat_i = 1
```

Interior MAP is:

```text
theta_i_MAP = (n_i + alpha_i - 1) / (N + alpha_0 - K)
```

Interior MAP goes to boundaries when `alpha_i < 1`, which is required for
sparse/homogeneous priors. Floor-constrained MAP keeps finite probabilities for
local move scoring. Positive effective weights:

```text
w_i = n_i + alpha_i - 1
```

share the non-floor mass. Nonpositive effective weights sit at `epsilon`.

Posterior mean remains optional:

```text
theta_i_postmean = (n_i + alpha_i) / (N + alpha_0)
```

Do not use MLE as default:

```text
theta_i_MLE = n_i / N
```

MLE gives zero probabilities for unseen categories and ignores prior structure.

MAP caveat: high `kappa` shrinks `theta_hat` toward the base distribution but
does not generally make broad/mixed finite datasets lower profiled cost than
pure/narrow datasets. It reduces the sparse-data advantage because the optimizer
still chooses the best cluster-specific identity parameter.

## Data Structures

### ClusterState

Current:

```text
counts[(pair_type_idx, bin_idx)] -> edge count
N -> total internal edges
atom_ids -> atoms in cluster
```

Add:

```text
species_counts[species_idx] -> atom count
edge_counts_by_type[pair_type_idx][bin_idx] -> edge count
edge_N_by_type[pair_type_idx] -> internal edge count for that pair type
```

Possible migration path:

1. Keep existing `counts[(pair_type_idx, bin_idx)]` for compatibility.
2. Add helper views/functions to split counts by pair type.
3. Later replace with explicit nested arrays/dicts if performance requires.

### Partition

Add global metadata:

```text
species_to_idx
atom_species_idx
alpha_species
alpha_edge_by_pair_type
kappa_species
kappa_edge
```

`partition_from_labels()` must populate species counts from atom labels and
frame species data. Current function only receives labels, edges, and
bin_scheme, so API must change.

Recommended API:

```python
partition_from_labels(
    atom_labels,
    atom_species,
    edges,
    bin_scheme,
    kappa_species,
    kappa_edge,
    ...
)
```

`atom_species` can be symbols or integer species ids.

## Scoring Functions

Create new module:

```text
src/graincluster/model/parameterized.py
```

Functions:

```python
posterior_mean(counts, alpha) -> probs
dirichlet_map_data_term(counts, alpha) -> float
species_data_term(species_counts, alpha_species) -> float
edge_data_term_by_type(edge_counts, alpha_edge) -> float
cluster_data_term(cluster, priors) -> float
self_information_parameterized(category, counts, alpha) -> float
```

Current profiled estimate:

```text
theta_hat_i = constrained_map(n_i, alpha_i, epsilon)
```

Current score:

```text
L(theta_hat) + L(counts | theta_hat)
```

with `L(theta_hat)` using statistical resolution as above.

## Move Scoring

Atom move changes two data components:

1. Species counts:

```text
src loses moved atom species
tgt gains moved atom species
```

2. Edge-bin counts:

For each incident edge:

```text
neighbor in src before move -> edge leaves src internal set
neighbor in target after move -> edge enters target internal set
otherwise cut status changes only
```

Recompute exact source and target data terms first. Frozen scoring can come
later. The new objective has more coupled species and edge terms; exact local
recompute is safer for first implementation.

Legacy implementation:

```text
score_move = exact data term(src,tgt before/after)
           + legacy cut delta
           + legacy gamma delta
```

Current implementation under the new prior framework:

```text
score_move = exact data delta
           + delta L_K
           + delta L_cut
```

with the new structural terms included only when their corresponding
hyperparameters are not `None`.

## Cluster Merge Scoring

For merge `A + B`:

```text
species_counts_merged = species_A + species_B
edge_counts_merged = internal_A + internal_B + cross_edges_between_A_B
```

Current new-framework score:

```text
delta = L_data(merged) - L_data(A) - L_data(B)
        + delta L_K
        + delta L_cut
```

Keep rule: scan cross edges before mutating atom labels.

## Tests

Add tests before replacing old behavior:

1. Dirichlet shape sanity:

```text
Beta(0.5,0.5) favors pure over 50/50 under parameterized score.
Beta(10,10) favors 50/50 over pure under parameterized score.
```

2. Posterior mean differs from MLE:

```text
unseen category has finite positive theta_hat.
```

3. Species term:

```text
pure C cluster lower cost than mixed cluster when kappa_species < 2.
large kappa_species shrinks pure-vs-mixed cost gap.
```

4. Edge term:

```text
narrow single-bin edge distribution lower cost when kappa_edge < B.
large kappa_edge shrinks narrow-vs-uniform cost gap.
```

5. Move consistency:

```text
score_move equals objective_after - objective_before for small synthetic graph.
```

6. Merge consistency:

```text
score_cluster_merge equals objective_after - objective_before.
```

7. Structural prior sanity:

```text
larger x_cut -> larger L_cut at fixed N,K.
larger N and K increase the weak-cut scale beta_NK.
cut_prior_beta0=None disables the cut prior path.
```

8. Regression:

```text
single-phase graph stays merged under reasonable gamma/beta.
two-domain graph splits under sparse species/edge priors.
```

## Migration Order

1. Add prior-building utilities for species and edge bins.
2. Add parameterized scoring module with standalone tests.
3. Extend `ClusterState` with species counts.
4. Update `partition_from_labels()` API and tests.
5. Implement exact `cluster_data_term()`.
6. Replace `Partition.objective()` data term.
7. Replace `score_move()` with exact local recompute.
8. Replace `score_cluster_merge()`.
9. Update optimizer tests.
10. Re-run full test suite.
11. Only after correctness, optimize with frozen local approximations.

## Open Choices

- Whether `base_species` should always be global composition or optionally
  uniform.
- Whether edge priors for linear bins should use uniform bins or empirical bin
  frequencies.
- Whether a later exact finite-precision code for `theta` is worth the added
  complexity beyond the current asymptotic two-part approximation.
- Whether `OTHER_ID` should pay species/edge data cost or remain free
  background. Current behavior keeps it free only from `gamma`, not data cost;
  revisit after first implementation.

## Profiling And Speedups

### Live Profiling Workflow

The SiC workflow script now supports live optimization profiling:

```bash
PYTHONPATH=/n/home12/lsteinberger/code/graincluster/src \
/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311/bin/python \
.agents/test/run_sic_tau0_normalized.py \
  --frame-index 0 \
  --tau-species 0 \
  --tau-edge 0 \
  --cluster-count-prior-mean 1 \
  --tau-k 0 \
  --cut-prior-beta0 1 \
  --profile-live
```

Current live profile output reports per-pass and per-round cumulative deltas for:

- `score_move`
- `exact_data_delta`
- `exact_data_delta_with_split`
- `source_after_removal_state`
- `cut_cost_delta_for_move`
- `data_term_from_parts`
- `apply_move`
- `split_cluster_if_disconnected`
- `score_cluster_merge`
- `apply_cluster_merge`
- `greedy_pass`
- `merge_sweep`
- `louvain_round`

This is intended as a first-pass engineering profiler, not a replacement for
`cProfile`, `perf`, or line-level profilers. It is designed to answer:

- whether runtime is dominated by move scoring or connectivity repair
- how often the split-aware path is actually taken
- whether merge scoring is negligible or significant
- whether exact local rescoring or structural deltas dominate runtime

### Candidate Speedups To Evaluate

No speedups in this section should be implemented blindly. Each should be
validated against the live profiler first.

#### 1. Skip Prior Terms That Are Provably Constant

Some hyperparameter settings can collapse whole prior contributions into
constants, which means they should not be recomputed in inner loops.

Examples:

- `tau_edge = 0` with quantile bins gives:

  ```text
  kappa_edge = B
  alpha_edge,b = 1
  ```

  for every edge bin. In that case the Dirichlet prior-density term

  ```text
  -sum_b (alpha_b - 1) log theta_b
  ```

  is identically zero. The Dirichlet normalizer is also constant with respect
  to counts for any nonempty block. That means the edge-prior contribution can
  often be elided in delta computations under this setting.

- `tau_species = 0` does **not** generically imply a uniform prior over species.
  It only does so when the base distribution itself is uniform. In the current
  model the species base is the global species composition, so this shortcut is
  only valid when the simulation composition is exactly balanced.

- `tau_k = None` already disables the cluster-count prior entirely.
- `cut_prior_beta0 = None` already disables the cut prior entirely.

The implementation goal should be to detect these constant-prior regimes once
up front and route the hot path around them.

#### 2. Replace Sparse Dict Filtering With Stable Per-Type Arrays

The current exact deltas still rebuild per-pair-type views from sparse dicts
repeatedly. A likely high-value refactor is:

- store per-pair-type dense count arrays directly in `ClusterState`
- maintain them incrementally on edge add/remove
- stop reconstructing dense arrays inside move scoring

This should reduce:

- repeated sparse key filtering
- repeated allocation of temporary dense arrays
- repeated Python-loop overhead in exact local deltas

#### 3. Numba For Numeric Inner Kernels

Numba is plausible for pure numeric kernels with stable array inputs:

- constrained MAP estimation
- multinomial code-length formulas
- Dirichlet prior-density evaluation
- cut-prior negative log density

Numba is **not** likely to help much on the current dict-heavy graph/state
manipulation path unless the data structures are first converted to stable
array-based layouts.

So the likely sequence is:

1. stabilize hot paths into arrays
2. then JIT the numeric kernels

not the reverse.

#### 4. Faster Connectivity Tests Before Full Split Reconstruction

The split-aware path is exact but expensive. A likely optimization is a cheap
pre-check that can reject most non-splitting moves before running full
component reconstruction.

Examples:

- articulation-point style local tests
- bounded BFS around the moved atom
- degree/topology shortcuts for obviously safe removals

This is likely important because exact split-aware scoring is currently the
main reason large SiC runs became much slower after the semantics fix.

#### 5. Cache Constant Block Metadata

Likely cheap wins:

- cache `alpha == 1` masks
- cache which prior terms are active at all
- cache constant Dirichlet normalizers per pair type
- cache `d_free = n_categories - 1`

These are small individually, but they are evaluated in very hot code.

#### 6. Profile-Guided Search Changes

If atom-level exact scoring remains dominant even after hot-path cleanup, the
next step may not be micro-optimization but reducing the number of candidate
moves evaluated:

- better candidate pruning
- stronger merge moves
- coarse-to-fine schedules
- delayed singleton proposals

This should only be considered after measurement, because it changes optimizer
search behavior rather than just speeding up fixed semantics.
