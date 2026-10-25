# The Deterministic Questionnaire

*Model-specification questionnaire for `irtree` (package 0.4). Source: "Week 7
thoughts" (J. M. Mehou), corrected against the code.*

## Purpose

The questionnaire is the deterministic middle layer between a user's intent and
the package code. Answering it fully specifies an IRTree analysis, and every
complete set of answers compiles to exactly one valid program. The AI overlay
does not write code: it explains, fills in answers, confirms them with the user,
and lets the questionnaire compile them.

Three ways to run an analysis use it differently:

| User | Input | Uses |
|---|---|---|
| **User 1**: knows roughly what they want, some IRTree exposure | the AI overlay | AI asks, explains, fills the questionnaire, confirms |
| **User 2**: knows exactly what they want, basic coding | the questionnaire directly | answers it; reads the compiled code |
| **User 3**: writes code | the package API | the questionnaire's compiled output is the same code |

An analysis has six groups of elements: the tree, the data, the node models,
the constraints across nodes, the priors, and the run settings. Each section
below covers one group.

## How each question is written

| Field | Meaning |
|---|---|
| **ID** | Stable key used by reachability rules, the compiler and the AI |
| **Prompt** | The plain-language question |
| **Description** | Short elaboration, only where the prompt is not self-explanatory |
| **Type** | single, multi, bool, integer, number, text, file, list, matrix |
| **Options** | The bounded answer set (or range) |
| **Default** | Used when unanswered; "required" means no default |
| **Asked when** | The reachability rule, an expression over earlier answers |
| **Maps to** | The code element the answer compiles into |

Every question either has a default or is required, so a form is complete once
the required questions on its path are answered.

## Order of asking

IDs are stable, but the order differs from the numbering in one place: the
covariate questions of Section 2 come **after** Section 4, because a covariate on
a shared parameter must be declared identically on every node that shares it.

```
Section 0 → Section 1 → Q2.0–Q2.2b → Section 4 → Q2.3–Q2.3a → Section 3 → Section 5 → Section 6
```

## What changed from the Week 7 draft

| ID | Change | Why (from the code) |
|---|---|---|
| Q0.2 | `linear_2node_shared_trait` replaced by `nested_3node_shared_trait`; each topology names its factory | Two binary nodes in sequence give 3 categories, not 4. `ers_tree()` has 3 nodes, nodes 2 and 3 share one trait |
| Q0.3e | **New**: shared traits for custom trees | Without it a custom tree cannot share a trait (`node_to_latent`) |
| Q1.5 | **New**: missing-value codes | Needed by `load_data`; the draft's closing note referred to it but no question asked it |
| Q1.6 | **New** (optional): covariate formulas | `load_data(formulas=...)` |
| Q2.1 / Q2.2 | Merged into one model question (Q2.1); Q2.2 retired | Link and model asked separately allow pairs that do not exist (probit GPCM, logit GRM); PCM was missing |
| Q2.2a | GRM removed from the list | GRM nodes have no discrimination |
| Q2.2b | Outcome count taken from the tree; only homogeneous vs mixed asked | The number of outcomes is fixed by Section 0 |
| Q2.3 | Asked after Section 4; choices propagate across shared traits and shared b | Identical declarations are required, else validation fails |
| Q3.1–Q3.2 | Only priors the sampler honours; θ mean fixed at 0; scope (all nodes / one node) added; thresholds, missing-covariate priors and slope variance added | `priors.set()` rejects the rest (0.4 fixes) |
| Q3.3b–c | Shifted Beta and bounded Normal kept for two-member blocks; a-correlation also covers logit nodes with slopes | Both are implemented; logit and GPCM nodes have slopes |
| Q4.2–Q4.3 | Slope sharing excludes GRM and one-slope nodes; guessing sharing is probit 3PL only | Constraint validation |
| Q5.1 | Minimum 1600 iterations in evidence mode for every model | Burn-in floor 600 + 1000 kept draws, Gibbs or not |
| Q5.2a | Meaning of `n_warmup` in each mode corrected | Cap for models with MH, burn-in for pure-Gibbs models |
| Q5.7 | "Cuts burn-in 2–5×" removed | Never measured |
| Section 6 | **New**: compile, validate, confirm | The AI's confirmation step needs the summaries and the validation result |

------------------------------------------------------------------------

## Section 0: The tree (`tree_spec`)

### Q0.1: How do you want to specify the tree?

- **Description:** Built-in picks a ready-made tree from the package (shape at
  Q0.2). Custom means you build the mapping matrix yourself (Q0.3a–e).
- **Type:** single
- **Options:** built-in · custom
- **Default:** required
- **Asked when:** always
- **Maps to:** nothing yet

### Q0.2: Which built-in tree topology?

- **Description:** Each option fixes the number of nodes, the mapping matrix,
  the number of categories, which nodes are polytomous and which share a trait.
- **Type:** single
- **Options:**

  | Option | Shape | Factory |
  |---|---|---|
  | `linear_2node_binary` | node 1 binary → node 2 binary; 3 categories | `disengagement_tree(binary=True)` |
  | `linear_2node_polytomous` | node 1 binary → node 2 three-outcome; 4 categories | `disengagement_tree(binary=False)` |
  | `nested_3node` | node 1 → node 2 on one branch, node 3 on the other; all binary; 4 categories | `distractor_tree()` |
  | `nested_3node_shared_trait` | as `nested_3node`, with nodes 2 and 3 driven by one trait; 4 categories, 2 traits | `ers_tree()` |

- **Default:** required
- **Asked when:** Q0.1 = built-in
- **Maps to:** the factory call (fixes R, M, the categories, `node_to_latent`)

### Q0.2a: Which labels? (optional)

- **Description:** Names only (nodes and categories) for readable output. It
  does not change the model.
- **Type:** single
- **Options** (filtered by Q0.2; generic and custom always available):
  - `linear_2node_binary`: disengagement (node 1 = rapid guess vs engaged;
    node 2 = correct vs incorrect)
  - `linear_2node_polytomous`: disengagement (node 1 = rapid guess vs engaged;
    node 2 = degree of correctness)
  - `nested_3node`: distractor (node 1 = semantically related vs not; node 2 =
    visually similar vs not, unrelated branch; node 3 = correct vs incorrect,
    related branch)
  - `nested_3node_shared_trait`: ERS (node 1 = endorsement direction; nodes 2
    and 3 = extremity on each branch)
  - generic (Node 1, Node 2, …; Category 1, …)
  - custom (enter names at Q0.4a)
- **Default:** the application labels of the factory
- **Asked when:** Q0.1 = built-in
- **Maps to:** `node_names`, `category_names` of the spec (generic: `None`)

### Q0.3a: How many observed categories?

- **Type:** integer ≥ 2
- **Default:** required
- **Asked when:** Q0.1 = custom
- **Maps to:** K, the rows of M

### Q0.3b: How many internal nodes?

- **Type:** integer ≥ 1
- **Default:** required
- **Asked when:** Q0.1 = custom
- **Maps to:** R, the columns of M

### Q0.3c: How many outcomes does each node have?

- **Description:** 2 = binary; more than 2 = polytomous. Fixes the values each
  column of the mapping matrix may contain.
- **Type:** integer ≥ 2, one per node
- **Default:** 2 for every node
- **Asked when:** Q0.1 = custom
- **Maps to:** the allowed entries of each column of M; which nodes are polytomous
- **Example:** node 1 binary, node 2 three-outcome → [2, 3]

### Q0.3d: Build the mapping matrix

- **Description:** One row per observed category, one column per node. A cell
  is the outcome produced at that node on the path to that category, or "—" if
  the path never reaches the node.
- **Type:** matrix K × R; each cell single choice
- **Options:** per cell (k, r): "—" or 0, …, (outcomes of node r) − 1
- **Default:** required
- **Asked when:** Q0.1 = custom
- **Validation:** rows distinct; every node reached by at least one row; each
  node's outcome values contiguous from 0
- **Maps to:** `IRTreeSpec(mapping=M, values_start=0)`, "—" → NaN
- **Example** (verbal aggression, 3 categories, 2 nodes): no = [0, —],
  perhaps = [1, 0], yes = [1, 1]

### Q0.3e: Do any nodes share a latent trait? (new)

- **Description:** Nodes that measure the same trait (e.g. extremity on both
  branches of an ERS tree) share one θ. Give each node a trait number; nodes
  with the same number share it. Their item parameters stay separate unless you
  also share them (Q4.1).
- **Type:** integer per node, 1-based, contiguous labels (1, …, D)
- **Default:** every node its own trait (1, 2, …, R)
- **Asked when:** Q0.1 = custom and R ≥ 2
- **Maps to:** `IRTreeSpec(node_to_latent=(…))`

### Q0.4: Name your nodes and categories? (optional)

- **Type:** bool
- **Default:** no
- **Asked when:** Q0.1 = custom, or Q0.2a = custom
- **Maps to:** nothing yet

### Q0.4a: Node names 1…R; category names 1…K

- **Type:** text, one per node and one per category
- **Default:** required
- **Asked when:** Q0.4 = yes
- **Maps to:** `node_names`, `category_names`

------------------------------------------------------------------------

## Section 1: Data (`ingest.load_data`)

Responses come as a file path (csv/tsv) or a pandas DataFrame, wide (one row per
person, one column per item) or long (one row per person-item cell).

### Q1.0: Do you have covariates?

- **Description:** Predictors beyond the responses, attached to persons (X),
  items (Q) or person-item cells (W).
- **Type:** bool
- **Default:** required
- **Asked when:** always
- **Maps to:** whether covariate arguments are passed to `load_data`

### Q1.1: How is your data organized?

- **Type:** single
- **Options:** (0) one long table with responses and covariates ·
  (1) one responses table plus one table per covariate kind
- **Default:** required
- **Asked when:** Q1.0 = yes
- **Maps to:** single-table mode (`roles=`) vs separate tables

### Q1.2: Your responses table

- **Description:** Wide: an id column and one column per item (headers become
  item ids). Long: person id, item id and response columns. The first category
  code is 0 or 1.
- **Type:** file/DataFrame + layout (wide, long, auto) + column names
  (person id; for long also item id and response) + first category code (0 or 1)
  + optional item columns (wide only, when the file holds other columns)
- **Default:** required (layout default auto)
- **Asked when:** Q1.0 = no, or Q1.1 = (1)
- **Maps to:** `load_data(responses, person_id=, item_id=, response_col=,
  responses_layout=, item_columns=, categories_start=)`

### Q1.3: Your covariate tables

- **Description:** One entry per table, tagged person (X, joins on person id),
  item (Q, joins on item id) or cell (W, long format, joins on both).
- **Type:** file/DataFrame per table + kind (X, Q, W)
- **Default:** required
- **Asked when:** Q1.1 = (1)
- **Maps to:** `load_data(person_covariates=, item_covariates=, cell_covariates=)`

### Q1.4: Your single long table

- **Type:** file/DataFrame + person id, item id and response columns + first
  category code + the columns that are person, item and cell covariates
- **Default:** required
- **Asked when:** Q1.1 = (0)
- **Maps to:** `load_data(responses, ..., roles={"X": [...], "Q": [...], "W": [...]})`

### Q1.5: Which codes mark missing values? (new)

- **Description:** `NaN` is always missing. List any other codes (e.g. 99). A
  missing response is excluded from the likelihood (valid under MCAR/MAR); a
  missing covariate value is estimated each iteration.
- **Type:** list of numbers, for responses and for covariates
- **Default:** (99,) for both
- **Asked when:** always (covariate codes only when Q1.0 = yes)
- **Maps to:** `missing_codes=`, `covariate_missing_codes=`

### Q1.6: Expand covariates with a formula? (new, optional)

- **Description:** A patsy formula per covariate kind, e.g. `C(gender) + escs`
  (dummy coding, interactions). Needs the `formula` extra.
- **Type:** text per kind (X, Q, W)
- **Default:** none (columns used as given)
- **Asked when:** Q1.0 = yes
- **Maps to:** `load_data(formulas={"X": ..., "Q": ..., "W": ...})`

------------------------------------------------------------------------

## Section 2: Node models (`node_models`)

One model per node. A node is specified on its own; sharing across nodes is
Section 4.

### Q2.0: Is the same node model used at every node?

- **Type:** bool
- **Options:** yes (specify once, applied to all R nodes) · no (per node)
- **Default:** no
- **Asked when:** R ≥ 2 and all nodes have the same number of outcomes
- **Maps to:** whether Q2.1–Q2.2b are asked once or per node

### Q2.1: Which model governs node r? (merged with the former Q2.2)

- **Description:** Probit models are sampled by exact Gibbs draws (fast, no
  tuning); logit models by Metropolis-Hastings. The link is a per-node choice.
- **Type:** single
- **Options** (filtered by node r's number of outcomes):

  | Option | Outcomes | Class (`make_node` key) |
  |---|---|---|
  | Rasch / 1PL, probit | 2 | `ProbitRaschNode` (`"rasch"`) |
  | 2PL, probit | 2 | `ProbitTwoPLNode` (`"2pl"`) |
  | 3PL, probit | 2 | `ProbitThreePLNode` (`"3pl"`) |
  | Rasch / 1PL, logit | 2 | `LogitRaschNode` (`"logit-rasch"`) |
  | 2PL, logit | 2 | `LogitTwoPLNode` (`"logit-2pl"`) |
  | 3PL, logit | 2 | `LogitThreePLNode` (`"logit-3pl"`) |
  | GRM (graded response), probit | ≥ 2 | `GRMNode` (`"grm"`) |
  | GPCM (generalized partial credit), logit | ≥ 2 | `GPCMNode` (`"gpcm"`) |
  | PCM (partial credit, a = 1), logit | ≥ 2 | `PCMNode` (`"pcm"`) |
  | custom | any | a user `CustomNode` subclass (code, not a choice) |

- **Default:** Rasch / 1PL, probit (binary nodes); GRM (polytomous nodes)
- **Asked when:** always, per node
- **Maps to:** `node_models[r] = make_node(key, ...)`
- **Custom:** the form records a reference to the user's class; writing the
  class (its `log_prob`) is Input 3 territory. A custom node is updated by MH.

### Q2.2a: One slope per item, or one slope for the whole node?

- **Description:** Per item lets each item discriminate differently; common
  forces a single slope at this node (parallel items).
- **Type:** single
- **Options:** per item · common
- **Default:** per item
- **Asked when:** Q2.1 ∈ {2PL, 3PL (probit or logit), GPCM}
- **Maps to:** `common_discrimination=False / True`

### Q2.2b: Do all items at node r have the same number of outcomes?

- **Description:** Homogeneous: every item has the node's number of outcomes.
  Mixed: give one count per item (e.g. [2, 2, 3, 3]); binary items follow the
  binary form of the model inside the same node. All polytomous items must have
  the same count, equal to the node's.
- **Type:** single + (mixed) integer per item
- **Options:** homogeneous · mixed
- **Default:** homogeneous
- **Asked when:** Q2.1 ∈ {GRM, GPCM, PCM}
- **Maps to:** `n_outcomes=` (from Section 0) or `pseudo_item_outcomes=`

### Q2.3: Which covariates act on node r? (asked after Section 4)

- **Description:** Covariates enter the node's linear predictor inside the
  slope: a(θ − b + Xβ + Qκ + Wγ).
- **Type:** multi
- **Options:** none · person (X) · item (Q) · person×item (W)
- **Default:** none
- **Asked when:** Q1.0 = yes
- **Propagation:** nodes sharing a trait (Q0.3e / Q0.2) are asked once for X;
  nodes sharing b (Q4.1) are asked once for Q. The answer is copied to every
  member, because one shared coefficient needs one design.
- **Maps to:** `person_covariates=`, `item_covariates=`, `pi_covariates=` on the node

### Q2.3a: Which columns enter node r?

- **Type:** multi, the ingested covariate names of the kinds chosen at Q2.3
- **Default:** all columns of that kind
- **Asked when:** a kind was chosen at Q2.3 (same propagation as Q2.3)
- **Maps to:** e.g. `person_covariates=["escs", "female"]`

------------------------------------------------------------------------

## Section 3: Priors and correlations (`priors`)

Every parameter has a default, so the section can be skipped. Resolution order:
node level → type level → package default.

**Package defaults:** θ ~ N(0, σ²) with σ² ~ InvGamma(1, 1) on dimensions whose
at least one node is slope-free (Rasch, GRM, PCM), and σ² = 1 where every node
has free slopes (identification); b ~ N(0, τ²) with τ² ~ InvGamma(1, 1) per b-group (also
the GRM thresholds); a ~ TruncNormal(1, 0.5) on (0, ∞); c ~ Beta(5, 17);
β, κ, γ ~ N(0, τ²) with τ² ~ InvGamma(1, 1); missing covariate values ~ N(0, 1);
θ correlations estimated, LKJ(1) over all latent dimensions; b and a correlations
fixed at 0.

**Only priors the sampler honours are offered.** `priors.set()` rejects anything
else, so a form answer can never run a different model than it says.

### Q3.0: Use the default priors for everything?

- **Type:** bool
- **Default:** yes
- **Asked when:** always
- **Maps to:** `build_default_priors(spec, node_models)`; if no, Q3.1–Q3.2

### Q3.1: Which priors do you want to change?

- **Type:** multi
- **Options** (each offered only where the parameter exists):
  - b: item difficulty (binary items)
  - thresholds / steps (GRM, GPCM, PCM; mixed nodes: binary and polytomous items separately)
  - a: slopes
  - a variance (estimate the spread of slopes)
  - c: guessing (3PL)
  - θ variance
  - β, κ, γ: covariate coefficients
  - missing covariate values (X, Q, W)
- **Default:** none
- **Asked when:** Q3.0 = no
- **Maps to:** which `priors.set(...)` calls are generated

### Q3.2: Prior for each selected parameter

- **Description:** A distribution in the variance slot means "estimate this
  variance with that prior".
- **Type:** scope (all nodes, or one node r) + family + hyperparameters
- **Options:**

  | Parameter | Allowed priors | Update |
  |---|---|---|
  | b, thresholds/steps, β, κ, γ | `NormalPrior(mean, variance)` without bounds; mean any value; variance a number or `InvGammaPrior(α, β)` | Gibbs |
  | θ variance | a number (fixed), `InvGammaPrior(α, β)` (Gibbs) or `HalfNormalPrior(scale)` on the SD (MH); θ's mean stays 0 | — |
  | a | `TruncNormalPrior(mean, variance, lower=0)` (Gibbs) · `LogNormalPrior(mean, variance)` without bounds (MH) | — |
  | a variance | `InvGammaPrior` or `HalfNormalPrior` in the slope prior's variance slot, with `force=True` (MH; not with one slope per node) | MH |
  | c | `BetaPrior(α, β)` | Gibbs (probit), MH (logit) |
  | missing X/Q/W | `NormalPrior(mean, variance)` with a fixed variance | Gibbs, or MH where a logit node uses it |

- **Default:** the package default for that parameter
- **Asked when:** that parameter was selected at Q3.1
- **Warning shown:** changing the θ variance on a dimension whose nodes all have
  free slopes removes the identification default (σ² = 1)
- **Maps to:** `priors.set(param, PriorSpec, node=None or r, force=...)`; keys
  `"b"`, `"b.binary"`, `"b.poly"`, `"a"`, `"c"`, `"theta"`, `"beta"`, `"kappa"`,
  `"gamma"`, `"X_missing"`, `"Q_missing"`, `"W_missing"`

### Q3.3: Are any parameters correlated across nodes or traits?

- **Description:** Three families can be correlated, each on its own: traits θ
  (across latent dimensions), difficulties b (across nodes), slopes a (across
  nodes with slopes). Correlated parameters stay separate; their covariance is
  estimated or fixed. Sharing (one common parameter) is Section 4.
- **Type:** bool
- **Default:** θ correlated (LKJ(1) over all latent dimensions, when D ≥ 2); b and a not correlated
- **Asked when:** ≥ 2 latent dimensions, or ≥ 2 binary nodes, or ≥ 2 nodes with slopes
- **Maps to:** if no, `FixedCorrelation(rho=0)` for b and a, and for θ only if the user
  chooses independent traits (`priors.set("theta.correlation", FixedCorrelation(rho=0))`)

### Q3.3a: Which families are correlated?

- **Type:** multi
- **Options:** θ (latent dimensions) · b (binary nodes) · a (nodes with per-item slopes)
- **Default:** required
- **Asked when:** Q3.3 = yes
- **Maps to:** which correlation keys receive a block and a prior

### Q3.3b: Correlation blocks for each selected family

- **Description:** A block is a set of two or more members correlated jointly;
  a block of size G gives a G×G matrix. θ blocks list latent-dimension numbers;
  b and a blocks list node numbers (a node stands for its sharing group).
  Blocks are disjoint; members not in a block stay independent.
- **Type:** list of blocks per family (each ≥ 2 one-based indices)
- **Options:**
  - θ: blocks of dimensions 1…D; empty = all D dimensions in one block
  - b: blocks of binary nodes; no two nodes from the same b-sharing group
  - a: blocks of nodes with per-item slopes; no two from the same a-sharing group
- **Default:** required
- **Asked when:** that family was selected at Q3.3a
- **Maps to:** `Constraints(theta_corr=[...], b_corr=[...], a_corr=[...])`

### Q3.3c: Prior on each correlation

- **Type:** single + hyperparameters, per family
- **Options:**

  | Prior | Block size | Update |
  |---|---|---|
  | `FixedCorrelation(rho=scalar or G×G matrix)` | any | fixed |
  | `LKJPrior(eta)` | any | MH (Fisher-z for G = 2, vine for G > 2) |
  | `InverseWishartPrior(scale, df)` | any | exact Gibbs (variances read from Σ) |
  | `NormalPrior(mean, variance, lower=-1, upper=1)` | 2 only | MH (Fisher-z) |
  | `BetaPrior(alpha, beta, lower=-1, upper=1)` | 2 only | MH (Fisher-z) |

- **Default:** θ `LKJPrior(eta=1)`; b and a `FixedCorrelation(rho=0)`
- **Asked when:** that family was selected at Q3.3a
- **Note:** one prior per family (it applies to all its blocks); for θ the
  two-member priors require D = 2
- **Maps to:** `priors.set("theta.correlation" | "b.correlation" | "a.correlation", PriorSpec)`

------------------------------------------------------------------------

## Section 4: Sharing across nodes (`constraints`)

Sharing ties a parameter across nodes to one estimated array. Covariate pooling
follows automatically: β is pooled across nodes sharing a trait, κ across nodes
sharing b; γ pooling is opt-in (Q4.4). Correlation is Section 3.

### Q4.0: Do any nodes share item parameters?

- **Type:** bool
- **Default:** no
- **Asked when:** R ≥ 2
- **Maps to:** the `*_shared` fields of `Constraints`; `None` without sharing or correlation

### Q4.1: Share difficulties b?

- **Type:** list of groups of 1-based node numbers, e.g. [[1, 3], [2, 4]]
- **Rule:** members have the same number of outcomes and the same per-item
  outcome counts (both homogeneous, or identical mixed vectors)
- **Default:** none
- **Asked when:** Q4.0 = yes
- **Maps to:** `Constraints(b_shared=[...])` (`share_nodes(1, 3)` is the same as `[1, 3]`)

### Q4.2: Share slopes a?

- **Type:** list of groups
- **Rule:** members have per-item slopes (2PL, 3PL, GPCM; not GRM or PCM, not
  one-slope-per-node nodes)
- **Default:** none
- **Asked when:** Q4.0 = yes and ≥ 2 such nodes exist
- **Maps to:** `Constraints(a_shared=[...])`

### Q4.3: Share guessing c?

- **Type:** list of groups
- **Rule:** members are probit 3PL nodes
- **Default:** none
- **Asked when:** Q4.0 = yes and ≥ 2 probit 3PL nodes exist
- **Maps to:** `Constraints(c_shared=[...])`

### Q4.4: Pool person×item covariate effects γ?

- **Type:** list of groups
- **Rule:** members declare the identical person×item columns (Q2.3a); this is
  asked again after Q2.3 if needed
- **Default:** none
- **Asked when:** Q4.0 = yes and ≥ 2 nodes declare person×item covariates
- **Maps to:** `Constraints(gamma_shared=[...])`

------------------------------------------------------------------------

## Section 5: Run settings (`irtree_fit`)

All defaulted. Evidence-mode warm-up is the default, so warm-up need not be set.

### Q5.1: Iterations per chain

- **Description:** In evidence mode every model needs room for a burn-in of at
  least 600 iterations (2·M·W) plus the guaranteed kept draws (1000 by default).
- **Type:** integer
- **Options:** ≥ 2·M·W + min_keep·thin_by (≈ 1600) in evidence mode; ≥ n_warmup + 1 in fixed mode
- **Default:** 2000
- **Asked when:** always
- **Maps to:** `n_iter`

### Q5.2: Warm-up mode

- **Type:** single
- **Options:** evidence (each MH step size freezes once its acceptance rate is
  stable; draws are kept once all have frozen) · fixed (adapt for exactly
  n_warmup iterations; for short diagnostic runs)
- **Default:** evidence
- **Asked when:** always
- **Maps to:** `warmup_mode`

### Q5.2a: n_warmup

- **Type:** integer ≥ 0
- **Options:**
  - evidence mode, model with MH blocks: optional cap on the warm-up (0 = automatic)
  - evidence mode, pure-Gibbs model: burn-in length, clamped to at least 600 (0 = 600)
  - fixed mode: number of adaptation/burn-in iterations (> 0)
- **Default:** 0
- **Asked when:** always (required > 0 when Q5.2 = fixed)
- **Maps to:** `n_warmup`

### Q5.2b: Minimum kept draws per chain

- **Type:** integer, or default
- **Default:** 1000 (raise for high-dimensional logit/GPCM blocks)
- **Asked when:** Q5.2 = evidence
- **Maps to:** `min_keep`

### Q5.3: Number of chains

- **Type:** integer ≥ 1 (≥ 2 for R-hat)
- **Default:** 4
- **Asked when:** always
- **Maps to:** `n_chains`

### Q5.4: Seed

- **Description:** Each chain gets its own child seed (`SeedSequence.spawn`).
- **Type:** integer, or none
- **Default:** 42
- **Asked when:** always
- **Maps to:** `seed`

### Q5.5: Thinning

- **Type:** integer ≥ 1
- **Default:** 1
- **Asked when:** always
- **Maps to:** `thin_by`

### Q5.6: Chain storage precision

- **Type:** single: float32 (half the memory) · float64
- **Default:** float32
- **Asked when:** always
- **Maps to:** `chain_dtype`

### Q5.7: Start item difficulties from the data?

- **Description:** Initializes b from each item's observed proportions instead
  of zero, which usually shortens burn-in.
- **Type:** bool
- **Default:** yes
- **Asked when:** always
- **Maps to:** `warm_start`

`categories_start` and the missing-value codes come from Section 1. `max_workers`
(parallel chains, default min(chains, CPUs)) and `progress` are optional
conveniences.

------------------------------------------------------------------------

## Section 6: Compile, check and confirm (new)

### Q6.1: Review the specification

- **Description:** The answers are compiled and checked before anything runs.
  The form shows the tree (`spec.summary()`), the priors (`priors.summary()`,
  with where each came from and the identification note), the constraints
  (`constraints.summary()`), the data summary (`summarize_prepared`), and the
  generated Python. The checks are the package's own: `IRTreeSpec` validation,
  `priors.set()` validation, `Constraints.validate(spec, node_models)`, and the
  covariate-selection checks of `resolve_covariate_indices`.
- **Type:** bool (confirm / go back)
- **Default:** required
- **Asked when:** always
- **Maps to:** `irtree_fit(...)` when confirmed

### Q6.2: Outputs after the run (optional)

- **Type:** multi
- **Options:** console verdict (always) · diagnostics table · HTML report
  (`fit.report`, first 50 parameters plotted, or a number, or all) · posterior
  table · logistic-equivalent estimates (`fit.to_logit()`) · WAIC / LOO
- **Default:** console verdict and posterior table
- **Asked when:** always
- **Maps to:** the calls after `irtree_fit`

------------------------------------------------------------------------

## The compiled program

A complete form compiles to one program, in this order (shown for a custom
two-node tree with person covariates):

```python
import irtree
from irtree import Constraints, NormalPrior

spec = irtree.IRTreeSpec(mapping=[[0, float("nan")], [1, 0], [1, 1]],      # Q0.3d
                         node_names=("engagement", "accuracy"))            # Q0.4a
data = irtree.load_data("responses.csv", spec=spec, person_id="id",       # Q1.2
                        person_covariates="persons.csv",                    # Q1.3
                        categories_start=1, missing_codes=(99,))            # Q1.2, Q1.5
nodes = [irtree.make_node("rasch", person_covariates=["escs"]),            # Q2.1, Q2.3a
         irtree.make_node("2pl")]
constraints = None                                                          # Section 4
priors = irtree.build_default_priors(spec, nodes)                           # Q3.0
priors.set("b", NormalPrior(mean=1.0, variance=1.0), node=1)                # Q3.2
fit = irtree.irtree_fit(data, spec, nodes, priors=priors, constraints=constraints,
                        n_iter=2000, n_chains=4, seed=42)                   # Section 5
fit.summary()                                                               # Q6.2
```

## Testing the "100%" guarantee

The finite-choice part of the form can be enumerated. The intended check is: for
every reachable path (or a large random sample of them), compile the answers
and require that every validation passes and that the generated code matches
the objects built directly. A machine-readable version of this document (the
questions as data, the reachability rules as expressions, one `compile(answers)`
function) makes that check automatic and keeps this document generated from the
same source.
