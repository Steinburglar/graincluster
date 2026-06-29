# Graincluster Implementation Plan

This document is the handoff for a new repository, `graincluster`.

It captures the current scientific target, the objective function, the
optimization strategy, the data model, and the implementation path needed to
start coding without rereading the whole discussion.

## 1. Purpose

`graincluster` is intended to find coherent phase pockets, interfaces, and
grain-boundary-like domains inside large atomistic simulations.

The target is not general atom-environment labeling in feature space. The
target is **configuration-space segmentation**: identify contiguous regions of
atoms whose internal edge statistics are low-information / low-disorder
relative to the rest of the system.

This is meant to complement, not replace, local structure analyzers such as
PTM or CNA.

## 2. Relationship To `graphcluster`

This should be a **new repository / extension package**, not a small patch to
the existing `graphcluster` codebase.

Reason:

- the objective is different
- the optimization backend is different
- the data model is close, but not identical
- the work is conceptually an MDL / entropy segmentation engine, not a Leiden
  tweak

`graincluster` should depend on `graphcluster` only where it makes sense:

- reuse `Frame`-like trajectory/graph data structures if convenient
- reuse graph-building utilities if they remain generic
- reuse IO and artifact writers only if they are not too coupled to the old
  runner

Do not force the current `graphcluster` runner architecture onto this problem.

## 3. Scientific Objective

The core idea is:

- build a graph over atoms
- assign a physical feature to each edge
- partition the atoms so that internal edge features have low information
- penalize cuts across strong edges
- optionally penalize complex species composition inside clusters

The partition should minimize a description-length style objective.

### 3.1 Edge information term

Edges are treated as observations with two parts:

- species pair type
- raw physical feature bin

The preferred initial feature is:

- raw interatomic distance, in physical units

Not CDF / percentile transformed values.

The main reason is that percentile transforms erase scale information that
matters scientifically. A liquid-like distribution and a crystal-like
distribution should not collapse to the same rank-shape merely because their
percentile histograms look similar.

### 3.2 Species handling

Species are not a nuisance variable. They are part of the information content.

The entropy model should therefore be **joint over species-pair type and raw
edge bin**.

This achieves:

- one-species reduction works automatically
- mixed-species clusters carry more information than pure species clusters
- different pair types do not artificially “help” each other by pooling raw
  values together

### 3.3 Boundary penalty

The cut penalty should be separate from the entropy term.

Use a smooth affinity:

$$
w_{ij} = \exp\left(-\frac{d_{ij}^2}{2\sigma^2}\right)
$$

or equivalently a “surprise” score:

$$
s_{ij} = -\log(w_{ij}+\epsilon) = \frac{d_{ij}^2}{2\sigma^2}
$$

This is not a probability model for the cut term. It is simply a smooth way
to make cuts across short/strong edges expensive and cuts across long/weak
edges cheap.

### 3.4 Final high-level objective

The current working objective is:

$$
L = \sum_C N_C H_C + \gamma K + \lambda \sum_{\mathrm{cut}} s_{ij}
$$

where:

- `C` indexes clusters
- `N_C` is the total number of internal edges in cluster `C`
- `H_C` is the joint entropy of pair type and raw edge bin in cluster `C`
- `K` is total cluster count
- `\gamma` is the model-complexity / resolution penalty
- `\lambda` controls the cut penalty
- `s_{ij}` is the cut cost for boundary edge `(i,j)`

The entropy term is the main MDL/data term.
The cluster-count term is the model-complexity term.
The cut penalty is the geometric regularizer.

## 4. Binning Strategy

The entropy term should use **raw-space linear bins**, not CDF bins.

### 4.1 Why not global CDF / percentile bins

CDF transform is wrong for this problem if the goal is to retain physical
scale. It makes the model insensitive to whether the raw edge distribution is
narrow like a crystal or broad like a liquid.

That is useful for comparing rank-shapes, but not for detecting disorder
changes under heating, melting, strain, or defect formation.

### 4.2 Raw bins

Use linear bins in raw physical units.

Recommended:

- choose bins per species pair
- use a robust physical range, e.g. clipped by low/high percentiles from a
  global reference set
- keep the bins fixed during optimization

Good default binning rule:

- take a global sample over the trajectory or over a training subset
- for each species pair, determine a robust range such as `[p1, p99]`
- divide that range linearly into `B` bins

This preserves physical scale and keeps the entropy term interpretable.

### 4.3 Pair-specific bin ranges

Do not force all species-pair types to share the same raw range unless a
specific chemistry justifies it.

Reasons:

- different pair types naturally live on different distance scales
- forcing one range can over-compress some pairs and under-resolve others
- pair-specific bins make the entropy term more meaningful

Use the same number of bins per pair type if you want the implementation to
stay simple.

## 5. Species Modeling

Species-pair type is part of the entropy model, not just a filter.

Let:

- `t` = pair type, e.g. `AA`, `AB`, `BB`
- `b` = raw distance bin

For a cluster `C`, maintain joint counts:

$$
n_{t,b}(C)
$$

This joint table captures both:

- species-pair composition
- distribution of raw edge lengths within each pair type

This is the right base model if you want mixed-species clusters to carry a
real information cost.

### 5.1 Why cluster-level normalization matters

Normalize the joint histogram over the **total internal edge mass in the
cluster**, not separately per pair type.

That choice is deliberate because it makes the cluster entropy sensitive to:

- how much the cluster mixes different pair types
- how broad the bond-length distribution is within each pair type

If each pair type were normalized separately, the model would stop charging
for species mixing. That would defeat part of the scientific goal.

### 5.2 One-species limit

If only one species exists, only one pair type exists. The species-pair part
collapses automatically and the model reduces to the single-species case,
modulo smoothing.

### 5.3 Dirichlet smoothing

Use Dirichlet / pseudocount smoothing so that absent categories do not give
`log(0)` singularities.

This is standard multinomial smoothing, not a special ad hoc trick.

## 6. Objective Form In Counts

For each cluster `C`, let the joint categories be indexed by `i = (t,b)`.

Maintain:

- `n_i(C)` = count in joint category `i`
- `N_C = Σ_i n_i(C)` = total internal edge count
- `α > 0` = pseudocount
- `M = T * B` = total number of joint categories

Smoothed counts:

$$
\tilde{n}_i(C) = n_i(C) + \alpha
$$

Smoothed total mass:

$$
\tilde{N}_C = N_C + \alpha M
$$

Smoothed probability:

$$
\tilde{p}_i(C) = \frac{\tilde{n}_i(C)}{\tilde{N}_C}
$$

Joint entropy:

$$
H_C = - \sum_i \tilde{p}_i(C)\log \tilde{p}_i(C)
$$

Data term:

$$
L_{\mathrm{data}} = \sum_C N_C H_C
$$

That is the current preferred entropy structure.

## 7. Optimization Philosophy

The implementation should use a **frozen-histogram / frozen-model move
evaluation** strategy.

That means:

- at a given optimization state, each cluster has a frozen empirical model
- a proposed move is scored against the current frozen model
- after accepting a move, counts are updated for the new state

This avoids recomputing full histograms from scratch on every proposal.

The move evaluation should be local:

- only incident edges of the moved vertex matter
- only the source and destination clusters need recomputation

This is the main performance requirement.

## 8. Exact Move Logic

Consider a proposed move of vertex `v` from cluster `A` to cluster `B`.

For each incident edge `e = (v,u)`:

- determine pair type `t_e`
- determine raw bin `b_e`
- determine whether the edge is internal to `A`, internal to `B`, or
  crossing after the move

Only the categories affected by those incident edges change.

### 8.1 Edge entropy contribution

For the frozen-model interpretation, the move score should be computed from
the current pre-move cluster model.

For a joint category `i`, the self-information is:

$$
I_C(i) = -\log \tilde{p}_i(C)
$$

where `\tilde{p}_i(C)` is the smoothed probability in the current cluster
state.

When an edge leaves a cluster, subtract its current self-information from
that cluster’s score. When an edge enters a cluster, add its current
self-information to that cluster’s score.

This is the correct local move approximation for a frozen empirical model.

### 8.2 Species composition term

If a separate species-composition term is included, treat it the same way:

- maintain a cluster-wise species multinomial
- use smoothed probabilities
- score atom movement with frozen self-information under the pre-move model

That term is optional, but if included it should be explicitly separated from
the edge joint entropy term.

### 8.3 Boundary term

The cut penalty is easier:

- edges from `v` to `A` become cut edges after moving
- edges from `v` to `B` stop being cut edges after moving

So the boundary delta is just the change in the set of cut edges incident to
`v`.

## 9. Species Composition Prior

There are two possible species-related information notions:

1. **Species composition as data term**
   - species labels inside a cluster have entropy
   - mixed species cost more bits to encode

2. **Species composition as model prior**
   - the multinomial parameters themselves may have a prior
   - this is a separate question from the observed data entropy

For the first implementation, keep the model simple:

- charge species composition through the data term
- do not yet add a separate exotic prior unless the data demands it

This keeps the interpretation clear:

- the entropy term is the lower bound on data encoding cost under the
  current partition
- the `\gamma K` term is the partition complexity penalty

If later needed, add a composition prior to prefer stoichiometric or
species-pure clusters.

## 10. First Implementation Scope

The first implementation should be as small as possible while still capturing
the intended science.

### 10.1 In scope

- single trajectory frame graph
- per-edge raw feature extraction
- species-pair joint histogram objective
- frozen-model move scoring
- cut penalty
- cluster-count penalty
- simple local move optimizer
- debug outputs and unit tests

### 10.2 Out of scope for first pass

- full streaming trajectory runner
- visualization pipeline
- lifecycle reports
- advanced post-processing
- multiple competing objective backends
- learned embeddings

Do not overbuild the first pass.

## 11. Recommended Package Layout

Suggested repository structure:

```text
graincluster/
  src/graincluster/
    __init__.py
    io/
    graph/
    features/
    model/
    optimizer/
    analysis/
    utils/
  tests/
  configs/
  docs/
```

### 11.1 Core modules

#### `io`
Trajectory reading and conversion to project-owned frame objects.

#### `graph`
Graph construction from frames, edge lists, and sparse adjacency.

#### `features`
Raw edge feature extraction:

- distance
- optionally energy-based features later
- species-pair typing

#### `model`
Entropy / MDL objective.

This module owns:

- joint counts
- smoothing
- probability evaluation
- cluster score calculation

#### `optimizer`
Move proposals, local score deltas, acceptance, and count updates.

#### `analysis`
Only later, when clustering is working:

- cluster summaries
- plots
- phase-pocket inspection

## 12. Data Model

Use small, explicit data structures.

### 12.1 Frame

Canonical per-frame data:

- atom positions
- species labels
- cell / periodicity
- optional metadata

### 12.2 Sparse graph

Per-frame sparse graph:

- nodes = atoms
- edges = local neighbors
- edge attributes = raw feature and pair type

### 12.3 Edge record

Each edge should expose:

- endpoints
- pair type
- raw feature value
- bin index
- whether it is currently internal or cut under a partition

### 12.4 Cluster state

Each cluster should maintain:

- `n_{t,b}` joint counts
- `N_C`
- entropy cache
- optional species counts
- optional species entropy cache
- cut-edge bookkeeping if useful

## 13. Initial Objective Variants

Implement in this order:

### Variant A: edge joint entropy only

This is the smallest viable objective:

$$
L = \sum_C N_C H_C + \lambda \sum_{\mathrm{cut}} s_{ij} + \gamma K
$$

where `H_C` is joint entropy over pair type and raw bin.

### Variant B: add species composition term

If the system needs explicit cost for mixed-species clusters, add a cluster-wise
species multinomial data term.

### Variant C: later composition prior

Only if needed, add an explicit prior over species composition fractions.

This should not be the first implementation.

## 14. Suggested Optimizer Strategy

Do not start by inventing a new global optimizer if a local-move strategy
suffices.

The first optimizer should be:

- greedy local move
- optional multi-pass refinement
- optional randomization or tabu if needed

Why:

- easiest to debug
- easiest to validate against objective math
- easiest to instrument for move deltas

If the objective later proves strong and local moves perform well, then a
Leiden-like multilevel refinement or custom C++ backend can be considered.

## 15. Why Not Start With Leiden

Leiden is not a great initial fit for this objective because:

- the current objective is not modularity/CPM
- custom entropy bookkeeping is not the native use case
- the needed move updates are not the same as standard edge-sum quality
  functions

If the implementation is a clean local-search engine, you can still later add
Leiden-like refinement ideas. But do not force Leiden semantics into the first
pass.

## 16. Validation Strategy

### 16.1 Unit tests

Write tests for:

- joint count updates
- entropy calculation from counts
- Dirichlet smoothing
- frozen move delta signs
- boundary penalty delta
- one-species reduction

### 16.2 Toy systems

Start with synthetic graphs:

- perfect single-species crystal-like graph
- two-species crystal with identical lattice geometry
- simple alloy-like mixed graph
- graph with a clear interface between two domains

### 16.3 Expected behaviors

For a good implementation:

- perfect crystal should produce low entropy
- liquid-like or mixed-disorder graph should produce higher entropy
- species-pure clusters should be cheaper than mixed clusters if the data
  supports that model
- a cut across strong short edges should be expensive
- one-species case should reduce correctly

## 17. Specific Questions Still To Decide

These are open but not blocking the first scaffold:

1. exact bin range per pair type
2. default number of bins
3. whether to add explicit species-composition term in v1
4. whether boundary penalty should be raw `-\log w` or linear `w`
5. whether move optimizer should be deterministic or stochastic
6. whether the first implementation should target one frame or streaming

The recommended answer for v1:

- pair-specific raw bins
- moderate fixed bin count
- no extra species prior beyond joint entropy unless tests show need
- `-\log w` or `d^2/(2σ^2)` cut penalty
- deterministic greedy local move
- one frame first, streaming later

## 18. Practical Coding Order

1. implement frame and graph data objects
2. implement species-pair edge typing
3. implement raw-bin assignment per pair type
4. implement joint count tables and smoothing
5. implement cluster entropy calculation
6. implement frozen move delta for one vertex move
7. implement greedy local optimizer
8. add unit tests on synthetic graphs
9. iterate on binning and penalty weights

## 18.1 Task List / Milestones

### Milestone 0: repository bootstrap

- create `graincluster` repository layout
- add packaging scaffold
- add README and implementation plan

### Milestone 1: core data model

- define frame object
- define sparse graph object
- define edge record with pair type, raw value, and bin index

### Milestone 2: feature extraction and binning

- implement species-pair typing
- implement per-pair raw-bin assignment
- implement robust bin-range selection

### Milestone 3: entropy model

- implement joint count tables `n_(t,b)`
- implement Dirichlet smoothing
- implement cluster entropy calculation
- implement one-species reduction tests

### Milestone 4: move scoring

- implement frozen self-information lookup
- implement vertex move delta for entropy term
- implement cut penalty delta
- optionally add species-composition delta if needed

### Milestone 5: optimizer

- implement greedy local move loop
- implement update of cluster counts after accepted move
- add convergence / no-improvement stopping

### Milestone 6: validation

- build synthetic graph tests
- verify expected behavior on pure crystal, mixed crystal, and interface toy
- tune alpha, gamma, lambda, and binning defaults

### Milestone 7: scientific extensions

- add streaming trajectory support
- add summaries and plots
- consider alternative boundary penalties or richer priors only after the
  first prototype works

## 19. Scientific Guardrails

Keep the following principles fixed:

- raw-space bins, not percentile bins
- species-pair identity is part of the information model
- cluster-level normalization over total cluster mass
- boundary penalty separate from entropy term
- smoothing to avoid infinite surprise
- frozen empirical counts for move scoring

## 20. Expected First Success Criterion

The first success is not “perfect scientific truth.”

The first success is:

- a small prototype that can separate clear synthetic phases
- with objective values that match intuition
- and move deltas that behave stably

Once that is true, the package can be extended to real grain-boundary and
phase-pocket datasets.
