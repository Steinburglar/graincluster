# graincluster

`graincluster` is an MDL-style atomistic segmentation engine. It finds
contiguous phase pockets, grain-boundary regions, and interfaces by optimizing a
partition of atoms in a fixed connectivity graph.

This is not a local environment labeler. The target is configuration-space
segmentation: clusters should have coherent atom composition and coherent
internal edge-length statistics.

## Current Scientific Model

The target model is now a **parameterized MAP model**, not a marginalized
evidence model.

For a fixed graph over `N` atoms, infer:

```text
M = partition of atoms into clusters
theta_C = cluster-specific identity parameters
```

Each cluster identity has two pieces:

```text
theta_species,C = multinomial probabilities over atom species
theta_edge,C,t  = multinomial probabilities over edge-length bins for pair type t
```

Generative story:

1. Choose a partition `M`.
2. For each cluster `C`, choose atom-type parameters `theta_species,C`.
3. Generate atom species in `C` from `theta_species,C`.
4. Edge pair types are then deterministic from endpoint atom species.
5. For each induced pair type `t`, choose edge-bin parameters `theta_edge,C,t`.
6. Generate internal edge-length bins from `theta_edge,C,t`.

Edge bins are still modeled as conditionally independent given cluster, pair
type, and `theta_edge,C,t`. This is an approximation: real bonds are
geometrically constrained. The approximation is kept because it gives a
tractable segmentation objective.

## Why Not Marginal Likelihood

The previous model treated cluster parameters as nuisance variables:

```text
P(D | M) = product_C integral P(D_C | theta_C) P(theta_C) dtheta_C
```

That is a valid Bayesian evidence model, but it is not the desired scientific
model here. After marginalization, every cluster has the same identity under
the generative process. Cluster score becomes only prior-predictive probability
of its data under one universal prior. A highly self-consistent phase is not
rewarded for having its own chosen identity; it is only rewarded if that data
pattern has high prior-predictive probability.

For this project, clusters should have explicit identities. A graphene pocket,
liquid pocket, grain interior, or interface should be represented by its own
composition and edge-length parameters. Therefore the model should not treat
those parameters as pure nuisance variables. The long-term target is:

```text
P(D, theta | M) = P(D | theta, M) P(theta | M)
```

not:

```text
P(D | M) = integral P(D | theta, M) P(theta | M) dtheta
```

At a high level, the partition prior can still be marginalized where that does
not destroy scientific meaning. In particular, a hyperparameter controlling the
expected number of clusters can be treated as nuisance and integrated out,
while the cluster identities `theta_C` should remain explicit.

In negative-log form, the target decomposition is:

```text
L(M, theta) =
    -log P(M)
  + sum_C [
      -log P(D_C | theta_C)
      -log P(theta_C)
    ]
```

This points toward a two-part MDL treatment: encode partition and cluster
identities, then encode data given those identities.

## Objective Shape

The current implementation has two objective paths.

Legacy path, kept only for compatibility:

```text
L_legacy = (1 - beta) * sum_C L_data(C)
         + beta * sum_cut s_ij
         + gamma * K
```

New prior-based path, used whenever any new structural prior hyperparameter is
set:

```text
L = sum_C L_data(C) + L_K + L_cut
```

where:

```text
L_K   = -log P(K)
L_cut = -log p(x_cut | N, K)
```

with `L_K` optional and `L_cut` optional. In the new framework, these are
additive prior terms. They are not mixed against the data term by an outer
`beta` coefficient.

The intended cluster data term is:

```text
L_data(C) =
    L_species(C, theta_species,C)
  + sum_t L_edge_bins(C, t, theta_edge,C,t)
```

where:

```text
L_species = -log P(species counts | theta_species,C)
            -log P(theta_species,C)

L_edge_bins = -log P(edge-bin counts for pair type t | theta_edge,C,t)
              -log P(theta_edge,C,t)
```

The current code uses explicit parameter estimates and an asymptotic two-part
parameter code. This is still an approximation, not an exact finite-precision
parameter code, but it is no longer a bare continuous-density energy.

## Priors

Both atom species and edge bins use Dirichlet priors over multinomial
parameters.

Write all Dirichlet parameters as:

```text
alpha_i = kappa * base_i
```

where:

- `base_i` is prior center / relative category weight.
- `kappa = sum_i alpha_i` controls concentration.
- If `alpha_i < 1`, prior favors sparse/corner distributions.
- If `alpha_i = 1`, prior is flat in that coordinate.
- If `alpha_i > 1`, prior favors interior/mixed distributions.

### Atom-Species Prior

For atom species:

```text
theta_species,C ~ Dirichlet(kappa_species * base_species)
```

Default:

```text
base_species_s = global atom fraction of species s in the simulation
```

`kappa_species` is a hyperparameter. It controls whether cluster compositions
are allowed to be homogeneous or pushed toward global composition.

For two species with `base = [0.5, 0.5]`:

```text
kappa_species < 2  -> bimodal / homogeneous clusters favored
kappa_species = 2  -> flat prior over composition
kappa_species > 2  -> theta_hat shrinks strongly toward global composition
```

For non-50/50 systems, `base_species` weights corners by global abundance.
With profiled MAP scoring, high `kappa_species` shrinks the advantage of
homogeneous counts but does not generally make mixed counts lower cost. The
optimizer still chooses the best cluster-specific `theta`.

### Edge-Bin Prior

For each species-pair type `t`:

```text
theta_edge,C,t ~ Dirichlet(kappa_edge * base_edge_t)
```

Default for quantile bins:

```text
base_edge_t,b = 1 / B_t
```

because quantile bins are fit from the global pair-type-specific CDF, so a
uniform base in quantile space corresponds to the global real-space edge-length
distribution for that pair type.

There is one edge prior per pair type, but for now all pair types share one
hyperparameter:

```text
kappa_edge
```

For `B_t` bins:

```text
kappa_edge < B_t  -> sparse/narrow edge-bin distributions favored
kappa_edge = B_t  -> flat prior over edge-bin multinomial parameters
kappa_edge > B_t  -> theta_hat shrinks strongly toward global-CDF-like bins
```

## Parameter Estimate Used During Optimization

Exact Dirichlet MAP:

```text
theta_i_MAP = (n_i + alpha_i - 1) / (N + alpha_0 - K)
```

works only for interior solutions where all `n_i + alpha_i > 1`. It goes to
boundaries when `kappa < K`, which is exactly the regime needed for
homogeneous/sparse clusters.

The implementation therefore uses a floor-constrained MAP estimate by default:

```text
theta_i_hat >= epsilon
sum_i theta_i_hat = 1
```

Positive effective weights `n_i + alpha_i - 1` share the non-floor mass. Empty
or prior-disfavored categories sit at `epsilon`. Posterior mean remains
available as an optional estimator:

```text
theta_i_postmean = (n_i + alpha_i) / (N + alpha_0)
```

Neither estimator is marginalized likelihood. Both produce explicit cluster
identity parameters used in the current profiled objective:

```text
L_data(C) = -log P(D_C | theta_hat_C) - log P(theta_hat_C)
```

Constrained MAP differs from the old MLE/plugin entropy path:

```text
theta_i_MLE = n_i / N
```

The floor keeps finite probabilities for unseen categories and incorporates the
prior shape through `alpha_i`.

The current score for each multinomial block is now an asymptotic two-part
code:

```text
L(counts, theta_hat) = L(counts | theta_hat) + L(theta_hat)
```

with:

```text
L(counts | theta_hat) = -log P(counts | theta_hat)
L(theta_hat) ≈ -log p(theta_hat) + (d / 2) log N
```

where `d` is the number of free parameters in that multinomial block and `N`
is the number of observations in that block. The multinomial coefficient and
Dirichlet normalizer are evaluated exactly with `lgamma`.

In profiled MAP scoring, high `kappa` shrinks `theta_hat`
toward the prior base and reduces the sparse-data advantage. It does not
generally turn a broad/mixed finite dataset into a lower-cost cluster than a
narrow/pure finite dataset. That behavior follows from choosing the
cluster-specific `theta` that minimizes description length.

## Statistical Resolution

To recover a true MDL/codelength interpretation while keeping explicit cluster
identities, `theta` should be encoded with a two-part code rather than with a
bare density term.

The core issue is that an exact real-valued parameter cannot be described with
finite bits. A density value `p(theta)` only becomes a code length after a
finite resolution is chosen.

If a one-dimensional parameter is quantized on a grid with spacing `delta`,
then a small cell around `theta` has probability mass approximately:

```text
P(cell around theta) ≈ p(theta) * delta
```

and the code length becomes:

```text
-log P(cell) ≈ -log p(theta) - log delta
```

For `d` free parameters:

```text
-log P(cell) ≈ -log p(theta) - d log delta
```

So yes: if `delta` is made arbitrarily fine, the parameter code grows even
though the data are unchanged. That means arbitrary user-chosen resolution is
not an acceptable notion of absolute information.

The standard fix is to use a statistical resolution rather than an arbitrary
grid spacing. For regular models with `N` effective observations, data only
identify parameters up to about:

```text
delta ~ N^(-1/2)
```

per free parameter. This yields the familiar asymptotic MDL/BIC-style
parameter penalty:

```text
L(theta_hat) ≈ -log p(theta_hat) + (d / 2) log N + const
```

This is the appropriate path for `theta_species,C` and `theta_edge,C,t`: keep
cluster identities explicit, and encode them with a finite-resolution/two-part
parameter code rather than a bare density score. This is the path currently
implemented.

## Structural Prior Direction

For structural complexity, the cluster-count prior is:

```text
K | lambda ~ Poisson(lambda)
lambda ~ Gamma(a, b)
```

with the Gamma prior expressed by mean and strength:

```text
E[lambda] = mu_K
strength = s_K
```

so that:

```text
a = s_K
b = s_K / mu_K
```

Then `lambda` is treated as a genuine nuisance hyperparameter and marginalized
out:

```text
P(K) = integral P(K | lambda) P(lambda) d lambda
```

and the cluster-count penalty becomes a proper information cost:

```text
L_K = -log P(K)
```

This is the right use of marginalization here:

- do **not** marginalize `theta_C`, because cluster identity is part of the
  scientific model;
- **do** marginalize `lambda`, because it only controls the prior over cluster
  count and is not itself an object of interest.

This gives two interpretable hyperparameters:

- `mu_K`: how many clusters are expected a priori,
- `s_K`: how strongly that expectation is enforced.

Current reparameterization:

```text
s_K = N_atoms * 10^(tau_K)
```

so `tau_K = 0` means the cluster-count prior strength scales directly with
system size.

Unlike the current `gamma * K` term, this produces a proper prior-predictive
surprise on `K`.

### Locality / Edge-Cut Prior

The locality prior lives at the same hierarchical level as the cluster-count
prior: it is a probabilistic description of one structural summary statistic of
the partition, not a replacement for the cluster data terms.

Define the total weighted cut:

```text
x_cut = sum_{(i,j) in boundary(M)} w_ij
```

where `w_ij` are the Gaussian-kernel edge weights already attached to graph
edges. The kernel bandwidth is treated as a fixed model parameter.

The current cut prior uses the fixed-shape special case of the
Exponential-Gamma construction:

```text
x_cut | lambda, N, K ~ Exponential(lambda)
lambda | N, K ~ Exponential(beta_NK)
```

This is equivalent to setting the Gamma prior shape to `1`. Because both levels
are exponential, this is the current "double exponential" formulation.

Marginalizing `lambda` gives a Lomax prior predictive with fixed shape `1`:

```text
p(x_cut | N, K) = beta_NK / (x_cut + beta_NK)^2
```

with scale:

```text
beta_NK = beta0 * N^(2/3) * K^(1/3)
```

The corresponding negative log density, in the numerically preferred form used
in the code, is:

```text
L_cut =
    log(beta_NK)
  + 2 log(1 + x_cut / beta_NK)
```

Interpretation:

- `x_cut = 0` is always the least surprising / simplest cut.
- Larger `N` and larger `K` increase the characteristic cut scale.
- Larger `beta0` weakens the locality prior by making larger cuts less
  surprising.
- `beta0 -> inf` is the weak-prior limit, but in code the cut prior should be
  disabled by setting `cut_prior_beta0 = None`, which also skips the
  computation entirely.

Because `p(x_cut | N, K)` is a continuous density, a literal codelength would
also require a finite cut resolution `Delta x`. If `Delta x` is held fixed
across candidate partitions, the additive `-log Delta x` term is constant and
can be dropped for optimization. The current objective therefore carries an
implicit, unspecified fixed `Delta x` for the cut statistic: it is assumed
constant across compared partitions and is not optimized or tuned.

One important limitation remains: `L_K` controls how many clusters there are,
and `L_cut` controls how much boundary they have, but neither term alone fully
codes the exact partition. They are additive prior factors over structural
summary statistics of `M`.

## Binning

Quantile binning is the preferred edge-bin scheme for this model:

```text
fit_bin_scheme_quantile(pair_values, n_bins)
```

Bins are fit separately per pair type and then frozen. Uniform edge prior in
quantile space means the prior predictive over real edge distances follows the
global reference distribution conditional on pair type.

Linear bins remain useful for physical interpretability, but the current model
discussion assumes pair-type-specific quantile bins.

## How To Run

The current parameterized workflow is easiest to run from the local
SiC-specific helper script:

```bash
PYTHONPATH=/n/home12/lsteinberger/code/graincluster/src \
/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311/bin/python \
.agents/test/run_sic_tau0_normalized.py
```

Example with only the new cluster-identity model active and no structural
priors:

```bash
PYTHONPATH=/n/home12/lsteinberger/code/graincluster/src \
/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311/bin/python \
.agents/test/run_sic_tau0_normalized.py \
  --frame-index 50 \
  --tau-species 0 \
  --tau-edge 0
```

Example with both new structural priors active:

```bash
PYTHONPATH=/n/home12/lsteinberger/code/graincluster/src \
/n/holylabs/kozinsky_lab/Users/lsteinberger/conda/envs/nequip311/bin/python \
.agents/test/run_sic_tau0_normalized.py \
  --frame-index 50 \
  --tau-species 0 \
  --tau-edge 0 \
  --cluster-count-prior-mean 1 \
  --tau-k 0 \
  --cut-prior-beta0 1
```

Current primary knobs in the new framework:

- `kappa_species`: Dirichlet concentration for atom species.
- `kappa_edge`: Dirichlet concentration for edge bins.
- `tau_species`: log-scale species prior concentration around the flat point
  via `kappa_species = N_species * 10^(tau_species)`.
- `tau_edge`: log-scale edge prior concentration around the flat point via
  `kappa_edge = N_bins * 10^(tau_edge)`.
- `cluster_count_prior_mean`: prior expected number of real clusters.
- `tau_k`: cluster-count prior strength via
  `strength_K = N_atoms * 10^(tau_k)`.
- `cut_prior_beta0`: global cut-scale parameter for the fixed shape-1 Lomax
  cut prior. `None` disables the cut prior and skips all related computation.
- `parameter_estimator`: use `constrained_map` by default; `posterior_mean` is diagnostic only.

Legacy `alpha`:

- `alpha` remains in some older helper paths only for backward compatibility.
- Do not treat `alpha` as the primary tuning knob for the current model.
- For the current model, the clean interpretation is `kappa_species` and `kappa_edge`.

Legacy structural knobs:

- `beta` and `gamma` are legacy-only outer energy weights.
- They are used only when no new structural prior knobs are active.
- Do not use them as the main interface for the new framework.

If you want to run a different SiC frame or use a different initialization,
copy the helper script and adjust:

```text
init = other | species | singleton
kappa_species
kappa_edge
cluster_count_prior_mean
tau_k
cut_prior_beta0
pair_cutoffs
bin count
```

For large SiC runs, the current exact local scoring is still expensive. The
helper script is correct but not optimized for parameter sweeps.

## Current State

Active context documents:

- [`agents.md`](./agents.md)
- [`CLAUDE.md`](./CLAUDE.md)

Deprecated historical plan:

- [`graincluster_implementation_plan.md`](./graincluster_implementation_plan.md)

The implementation currently contains graph construction, binning, partition
state, asymptotic two-part multinomial parameter coding, additive `K + cut`
structural priors, greedy atom moves, Louvain-style cluster merges, and tests.

The main modeling next steps are:

1. validate and tune the additive `K + cut` structural prior on real systems;
2. decide whether `OTHER_ID` should be structurally special beyond being
   excluded from `K`;
3. revisit whether a fuller prior on exact partition geometry is needed beyond
   the current summary-statistic priors.

## Implementation Plan

Detailed implementation plan:

- [`parameterized_model_plan.md`](./parameterized_model_plan.md)
