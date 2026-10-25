"""
irtree.tree_spec
================
Layer 1 — IRTree specification via mapping matrices.

The mapping matrix **M** is the single data structure from which every
downstream computation (pseudo-responses, relevance masks, index sets)
is derived.  It is a [K × R] array where:

* K  = number of observed response categories
* R  = number of internal decision nodes in the tree

Entry ``M[k, r]`` is the pseudo-response value at node *r* when observed
category *k* is reached.  ``NaN`` means node *r* is not on the path to
category *k*.

INDEXING CONVENTION
-------------------
**Internally, all indices are 0-based** — arrays are indexed with
``r = 0, ..., R-1`` and ``k = 0, ..., K-1``.  This matches Python/NumPy
convention and every downstream module uses these indices directly.

**User-facing output is 1-based** — ``summary()``, ``__repr__()``, and
error messages display nodes as "Node 1", "Node 2", etc. and latent
dimensions as θ₁, θ₂, etc.  This matches the notation in the IRTree
literature, making it easier for researchers to cross-reference between
package output and their model diagrams.

**Pseudo-response values in the mapping matrix** default to 0-based
(e.g. ``{0, 1}`` for binary, ``{0, 1, 2}`` for trichotomous).  Users
who prefer 1-based coding (e.g. ``{1, 2}`` or ``{1, 2, 3}``) can pass
``values_start=1`` to ``IRTreeSpec``, and the constructor will subtract 1
from all non-NaN entries automatically.

The boundary is clean: everything *stored* is 0-based, everything
*printed* is 1-based.  No downstream module ever needs to subtract 1.

Design principles
-----------------
1.  **Single source of truth.**  The mapping matrix fully encodes the tree
    topology.  No separate adjacency list or tree object is maintained.
2.  **Immutable after construction.**  The frozen dataclass validates the
    mapping once, in ``__post_init__``; downstream modules can rely on the
    spec being valid.
3.  **Derived quantities are computed properties**, evaluated on demand from
    the mapping matrix and reused where the caller stores the result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Validation helper (module-level, called before IRTreeSpec is fully built)
# ---------------------------------------------------------------------------

def _check_cross_node_consistency(M: np.ndarray, values_start: int) -> None:
    """Detect mixed 0-based / 1-based pseudo-response conventions across nodes.

    For each node, determines whether its values are contiguous from 0 or
    from 1.  If some nodes start at 0 and others start at 1, the user
    mixed conventions.  Neither ``values_start=0`` nor ``values_start=1``
    can fix this — the user needs to make their matrix consistent first.

    Called on the **original** (pre-shift) matrix in ``__post_init__``,
    before any values_start adjustment is applied.

    Parameters
    ----------
    M : np.ndarray, shape [K, R]
        The mapping matrix as the user provided it.
    values_start : int
        The user's declared convention (0 or 1).
    """
    R = M.shape[1]
    node_bases = {}  # r -> detected base (0 or 1), only for nodes with
                     # contiguous values from a clear base

    for r in range(R):
        vals = M[:, r][~np.isnan(M[:, r])]
        if len(vals) == 0:
            continue
        unique = np.sort(np.unique(vals))
        # Check that values are integers
        if not np.all(unique == np.floor(unique)):
            continue  # will be caught by _validate later
        unique_int = unique.astype(int)
        if len(unique_int) < 2:
            # A single unique value (e.g., {0} or {1}) is ambiguous —
            # we can't tell what convention the user intended.  Skip.
            continue
        min_val = int(unique_int[0])
        # Check if contiguous from min_val
        expected = np.arange(min_val, min_val + len(unique_int))
        if np.array_equal(unique_int, expected) and min_val in (0, 1):
            node_bases[r] = min_val

    if len(node_bases) < 2:
        return  # can't detect inconsistency with 0 or 1 node

    detected_bases = set(node_bases.values())
    if len(detected_bases) <= 1:
        return  # all nodes agree — consistent

    # Mixed: some nodes start at 0, others at 1
    zero_nodes = [r for r, b in node_bases.items() if b == 0]
    one_nodes = [r for r, b in node_bases.items() if b == 1]

    def _label(r):
        return f"Node {r + 1}"

    zero_desc = ", ".join(_label(r) for r in zero_nodes)
    one_desc = ", ".join(_label(r) for r in one_nodes)

    raise ValueError(
        f"Inconsistent pseudo-response coding across nodes.  "
        f"Some nodes use 0-based values ({zero_desc}) while others "
        f"use 1-based values ({one_desc}).  "
        f"All nodes must use the same convention.  Either rewrite all "
        f"values to start from 0 (and use values_start=0, the default) "
        f"or rewrite all values to start from 1 (and use values_start=1)."
    )


# ---------------------------------------------------------------------------
# IRTreeSpec dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IRTreeSpec:
    """Immutable specification of an IRTree model.

    Parameters
    ----------
    mapping : np.ndarray
        [K × R] mapping matrix.  Entry ``M[k, r]`` is the pseudo-response
        value at node *r* (0-based) for observed category *k*, or ``NaN``
        if node *r* is not on the path to category *k*.  Pseudo-response
        values are non-negative integers (stored as float to allow NaN).
    category_names : list[str] or None
        Optional length-K list of human-readable category labels.
    node_names : list[str] or None
        Optional length-R list of human-readable node labels.
    node_to_latent : list[int] or None
        Length-R mapping from node index to latent-dimension index.
        **1-based in user input** — e.g. ``[1, 2, 2]`` means the second
        and third nodes share latent dimension 2 (θ₂).  Internally
        converted to 0-based on construction for array indexing.
        Default (None) means each node has its own latent dimension.
        Values must be ≥ 1 because 0 is not a valid 1-based dimension
        index; passing 0 raises ``ValueError``.
    values_start : int (0 or 1), default 0
        Whether pseudo-response values in the mapping matrix start at 0
        or 1.  When ``values_start=1``, the constructor subtracts 1 from
        all non-NaN entries in the mapping matrix, converting them to the
        internal 0-based representation.  This lets users specify the tree
        in whatever convention is natural — e.g. writing ``[1, 2, 3]``
        for a trichotomous node instead of ``[0, 1, 2]``.

        Example::

            # 0-based (default):
            M = np.array([[1, np.nan], [0, 0], [0, 1]])
            spec = IRTreeSpec(mapping=M)

            # 1-based (equivalent result):
            M = np.array([[2, np.nan], [1, 1], [1, 2]])
            spec = IRTreeSpec(mapping=M, values_start=1)

    Attributes (derived, cached)
    ----------------------------
    n_categories, n_nodes, n_latent_dims, outcomes_per_node
    """

    mapping: np.ndarray
    category_names: tuple[str, ...] | None = None
    node_names: tuple[str, ...] | None = None
    node_to_latent: tuple[int, ...] | None = None
    values_start: int = 0

    # -- post-init validation -----------------------------------------------

    def __post_init__(self) -> None:
        # Coerce mapping to float64 ndarray (allows NaN)
        m = np.asarray(self.mapping, dtype=np.float64)
        if m.ndim != 2:
            raise ValueError(
                f"Mapping matrix must be 2-D [K × R]; got shape {m.shape}."
            )
        # frozen=True forbids direct assignment; use object.__setattr__
        object.__setattr__(self, "mapping", m)

        K, R = m.shape

        # --- values_start: shift pseudo-response values to 0-based --------
        vs = self.values_start
        if vs not in (0, 1):
            raise ValueError(
                f"values_start must be 0 or 1; got {vs}."
            )

        # --- Cross-node consistency check (on the original, pre-shift matrix) --
        # Detect when the user mixed 0-based and 1-based conventions across
        # nodes.  For each node, determine the minimum non-NaN value.  If
        # values are contiguous from that minimum, the minimum is the "base"
        # for that node.  If some nodes have base 0 and others have base 1,
        # the user mixed conventions — neither values_start=0 nor
        # values_start=1 will work, and we should say so explicitly.
        _check_cross_node_consistency(m, vs)

        if vs == 1:
            # Subtract 1 from all non-NaN entries to convert to 0-based.
            # After this, the mapping matrix is in the same internal form
            # as if the user had written 0-based values directly.
            # values_start is preserved (not reset to 0) so that user-facing
            # methods like summary() and category_path() can display values
            # in the user's original convention.
            shifted = m.copy()
            mask = ~np.isnan(shifted)
            shifted[mask] -= 1
            object.__setattr__(self, "mapping", shifted)
            m = shifted

        # Coerce mutable lists to tuples for frozen dataclass
        if self.category_names is not None:
            names = tuple(self.category_names)
            if len(names) != K:
                raise ValueError(
                    f"category_names length {len(names)} != K={K}."
                )
            object.__setattr__(self, "category_names", names)

        if self.node_names is not None:
            names = tuple(self.node_names)
            if len(names) != R:
                raise ValueError(
                    f"node_names length {len(names)} != R={R}."
                )
            object.__setattr__(self, "node_names", names)

        if self.node_to_latent is not None:
            ntd = tuple(self.node_to_latent)
            if len(ntd) != R:
                raise ValueError(
                    f"node_to_latent length {len(ntd)} != R={R}."
                )
            if any(d < 1 for d in ntd):
                raise ValueError(
                    f"node_to_latent values must be ≥ 1 (1-based latent "
                    f"dimension indices); got {ntd}."
                )
            # Require contiguous 1-based labels: the distinct dimensions must
            # be exactly {1, 2, ..., D} with no gaps. The label is only a
            # grouping tag — only the *pattern* of which nodes share a
            # dimension carries meaning, not the specific integers — so a
            # gapped labeling like (1, 1, 3) is just a mislabeling of
            # (1, 1, 2). Downstream code (dim_to_nodes, the per-dimension
            # theta storage, and beta pooling) indexes dimensions by dense
            # position 0..D-1, so a gap would otherwise fail later with an
            # opaque IndexError. Reject it here, at construction, with a
            # message the user can act on.
            distinct = sorted(set(ntd))
            expected = list(range(1, len(distinct) + 1))
            if distinct != expected:
                missing = sorted(set(expected) - set(distinct))
                raise ValueError(
                    f"node_to_latent labels must be contiguous, using every "
                    f"integer from 1 to D with no gaps; got distinct labels "
                    f"{distinct} (missing {missing}). The label is only a "
                    f"grouping tag, so relabel consecutively — e.g. write "
                    f"(1, 1, 2) instead of (1, 1, 3) to mean 'nodes 1 and 2 "
                    f"share a dimension, node 3 has its own'."
                )
            # Store 0-based internally: user's θ1 → index 0
            object.__setattr__(self, "node_to_latent", tuple(d - 1 for d in ntd))

        self._validate()

    # -- display helpers (single source for 1-based labels) -----------------

    def _node_label(self, r: int) -> str:
        """1-based display label for node at position r (0-based).

        Uses node_names if available; otherwise 'Node_{r+1}'.
        All user-facing output goes through this method so that the
        0-based → 1-based mapping happens in exactly one place.
        """
        if self.node_names:
            return self.node_names[r]
        return f"Node_{r + 1}"

    def _cat_label(self, k: int) -> str:
        """1-based display label for category at position k (0-based)."""
        if self.category_names:
            return self.category_names[k]
        return f"cat_{k + 1}"

    # -- input resolution (user-facing → 0-based) --------------------------

    def _resolve_category(self, category) -> int:
        """Convert a user-facing category reference to a 0-based index.

        Accepts:
          - str: matched against category_names (exact match required).
          - int: interpreted as 1-based position (1 = first category).

        Raises ValueError with a helpful message on mismatch.
        """
        if isinstance(category, str):
            if self.category_names is None:
                raise ValueError(
                    f"Cannot look up category by name ('{category}') — "
                    f"no category_names were provided.  "
                    f"Use a 1-based integer: 1 through {self.n_categories}."
                )
            if category not in self.category_names:
                raise ValueError(
                    f"Category '{category}' not found.  "
                    f"Available: {list(self.category_names)}."
                )
            return self.category_names.index(category)

        k = int(category)
        if k < 1:
            raise ValueError(
                f"Category indices are 1-based; got {k}.  "
                f"Use 1 for the first category."
            )
        if k > self.n_categories:
            raise ValueError(
                f"Category {k} is out of range; "
                f"this tree has {self.n_categories} categories (1–{self.n_categories})."
            )
        return k - 1

    def _resolve_node(self, node) -> int:
        """Convert a user-facing node reference to a 0-based index.

        Accepts:
          - str: matched against node_names (exact match required).
          - int: interpreted as 1-based position (1 = first node).

        Raises ValueError with a helpful message on mismatch.
        """
        if isinstance(node, str):
            if self.node_names is None:
                raise ValueError(
                    f"Cannot look up node by name ('{node}') — "
                    f"no node_names were provided.  "
                    f"Use a 1-based integer: 1 through {self.n_nodes}."
                )
            if node not in self.node_names:
                raise ValueError(
                    f"Node '{node}' not found.  "
                    f"Available: {list(self.node_names)}."
                )
            return self.node_names.index(node)

        r = int(node)
        if r < 1:
            raise ValueError(
                f"Node indices are 1-based; got {r}.  "
                f"Use 1 for the first node."
            )
        if r > self.n_nodes:
            raise ValueError(
                f"Node {r} is out of range; "
                f"this tree has {self.n_nodes} nodes (1–{self.n_nodes})."
            )
        return r - 1

    # -- core validation ----------------------------------------------------

    def _validate(self) -> None:
        """Run all structural checks on the mapping matrix.

        Checks performed (from first principles):

        1. Every category must pass through at least one node — otherwise
           the category has probability 1 regardless of parameters, which
           is degenerate.
        2. Every node must be relevant to at least two categories — a node
           visited by only one category contributes no information.
        3. At every relevant node, pseudo-response values must be
           non-negative integers — they index into the node's outcome set.
        4. At every node, the set of pseudo-response values must be
           contiguous starting from 0 — {0, 1, ..., M_r - 1}.
        5. Every category must have a unique path through the tree —
           no two rows of M may be identical after replacing NaN with a
           fixed sentinel value.  (Two rows that differ only in their
           NaN pattern are still distinguishable, but two rows that are
           element-wise equal after filling would be indistinguishable
           to the model.)
        """
        M = self.mapping
        K, R = M.shape

        if K < 2:
            raise ValueError(f"Need at least 2 categories; got K={K}.")
        if R < 1:
            raise ValueError(f"Need at least 1 node; got R={R}.")

        # Check 1: every category passes through ≥ 1 node
        for k in range(K):
            if np.all(np.isnan(M[k, :])):
                raise ValueError(
                    f"Category '{self._cat_label(k)}' (row {k}) has no "
                    f"relevant nodes (all NaN)."
                )

        # Check 2: every node is relevant to ≥ 2 categories
        for r in range(R):
            n_relevant = int(np.sum(~np.isnan(M[:, r])))
            if n_relevant < 2:
                raise ValueError(
                    f"{self._node_label(r)} (column {r + 1} of M) is "
                    f"relevant to only {n_relevant} "
                    f"categor{'y' if n_relevant == 1 else 'ies'}; need ≥ 2."
                )

        # Check 3 & 4: pseudo-response values are non-negative integers,
        # contiguous from 0
        for r in range(R):
            vals = M[:, r][~np.isnan(M[:, r])]
            if not np.all(vals == np.floor(vals)):
                raise ValueError(
                    f"{self._node_label(r)}: pseudo-response values must "
                    f"be integers; got {vals}."
                )
            if np.any(vals < 0):
                raise ValueError(
                    f"{self._node_label(r)}: pseudo-response values must "
                    f"be ≥ 0; got {vals}."
                )
            unique_sorted = np.sort(np.unique(vals)).astype(int)
            expected = np.arange(len(unique_sorted))
            if not np.array_equal(unique_sorted, expected):
                # Detect the common mistake: values are contiguous from 1
                # instead of 0 (user thinking in 1-based).
                expected_from_1 = np.arange(1, len(unique_sorted) + 1)
                if np.array_equal(unique_sorted, expected_from_1):
                    hint = (
                        "  It looks like your values start at 1 instead "
                        "of 0.  If you are using 1-based pseudo-response "
                        "coding, pass values_start=1 to IRTreeSpec."
                    )
                else:
                    hint = ""
                raise ValueError(
                    f"{self._node_label(r)}: pseudo-response values must "
                    f"be contiguous from 0; got {unique_sorted.tolist()}."
                    f"{hint}"
                )

        # Check 5: unique paths
        # Replace NaN with a sentinel that won't collide with valid values
        sentinel = -999.0
        M_filled = np.where(np.isnan(M), sentinel, M)
        unique_rows = set()
        for k in range(K):
            row_key = tuple(M_filled[k, :])
            if row_key in unique_rows:
                raise ValueError(
                    f"Duplicate path in mapping matrix at category "
                    f"'{self._cat_label(k)}' (row {k})."
                )
            unique_rows.add(row_key)

    # -- basic properties ---------------------------------------------------

    @property
    def n_categories(self) -> int:
        """K — number of observed response categories."""
        return self.mapping.shape[0]

    @property
    def n_nodes(self) -> int:
        """R — number of internal decision nodes."""
        return self.mapping.shape[1]

    @property
    def n_latent_dims(self) -> int:
        """Number of unique latent dimensions.

        Equals R when every node has its own latent variable (the default).
        Less than R when nodes share a latent variable via *node_to_latent*.
        """
        if self.node_to_latent is None:
            return self.n_nodes
        return len(set(self.node_to_latent))

    @property
    def dim_to_nodes(self) -> list[list[int]]:
        """Map each latent dimension to the nodes that use it.

        Returns a list of length ``n_latent_dims``.  Entry *d* is the
        sorted list of 0-based node indices whose person parameters are
        drawn from latent dimension *d*.

        When ``node_to_latent`` is None (the default, every node has its
        own dimension), the result is ``[[0], [1], ..., [R-1]]``.

        This is the **canonical source** for the dimension-to-nodes
        mapping.  ``mcmc_sampler.py`` reads this property to build the
        pooled theta Gibbs update across nodes that share a latent
        dimension; ``diagnostics.py`` uses it to group chain columns by
        latent dimension in the posterior summary.

        Examples
        --------
        >>> spec = ers_tree()
        >>> spec.dim_to_nodes   # nodes 2 & 3 (1-based) share latent dim 1 (0-based)
        [[0], [1, 2]]

        >>> spec = distractor_tree()
        >>> spec.dim_to_nodes   # each node has its own dimension
        [[0], [1], [2]]
        """
        if self.node_to_latent is None:
            return [[r] for r in range(self.n_nodes)]
        D = self.n_latent_dims
        groups: list[list[int]] = [[] for _ in range(D)]
        for r, d in enumerate(self.node_to_latent):
            groups[d].append(r)
        return groups

    @property
    def outcomes_per_node(self) -> list[int]:
        """Number of distinct pseudo-response outcomes at each node.

        Returns a list of length R.  Entry r is the number of distinct
        non-NaN values in column r of the mapping matrix:
        ``max(non-NaN values) + 1`` (since values are 0-based internally).

        Binary nodes return 2; a trichotomous node returns 3; and so on.
        """
        result = []
        for r in range(self.n_nodes):
            vals = self.mapping[:, r][~np.isnan(self.mapping[:, r])]
            result.append(int(np.max(vals)) + 1)
        return result

    # -- path queries (user-facing: accepts name or 1-based int) -------------

    def category_path(self, category) -> dict[str, object]:
        """Show the path through the tree for one observed category.

        Returns a dict keyed by **node display labels** (names or
        Node_1, Node_2, …).  Relevant nodes show their integer pseudo-
        response value; irrelevant nodes show ``"NA"``.  This gives
        the user a complete picture of which nodes are on the path.

        Parameters
        ----------
        category : str or int
            Category name (if category_names were provided) or 1-based
            position (1 = first category).

        Returns
        -------
        dict[str, int | str]
            Keys are node labels; values are pseudo-response integers
            or ``"NA"`` for nodes not on this category's path.

        Examples
        --------
        >>> spec = disengagement_tree()
        >>> spec.category_path("rapid")
        {'disengagement': 1, 'ability': 'NA'}
        >>> spec.category_path(2)   # 1-based: second category
        {'disengagement': 0, 'ability': 0}
        """
        k = self._resolve_category(category)
        row = self.mapping[k, :]
        vs = self.values_start
        return {
            self._node_label(r): (int(row[r]) + vs if not np.isnan(row[r]) else "NA")
            for r in range(self.n_nodes)
        }

    def relevant_categories(self, node) -> list[str]:
        """List the categories that pass through a given node.

        Parameters
        ----------
        node : str or int
            Node name (if node_names were provided) or 1-based position
            (1 = first node).

        Returns
        -------
        list[str]
            Category labels (names or cat_1, cat_2, …) in order.

        Examples
        --------
        >>> spec = disengagement_tree()
        >>> spec.relevant_categories("ability")
        ['incorrect', 'correct']
        >>> spec.relevant_categories(1)   # 1-based: first node
        ['rapid', 'incorrect', 'correct']
        """
        r = self._resolve_node(node)
        indices = np.where(~np.isnan(self.mapping[:, r]))[0]
        return [self._cat_label(int(k)) for k in indices]

    def is_binary_node(self, node) -> bool:
        """True if the given node has exactly 2 outcomes.

        Parameters
        ----------
        node : str or int
            Node name or 1-based position.
        """
        r = self._resolve_node(node)
        return self.outcomes_per_node[r] == 2

    def all_nodes_binary(self) -> bool:
        """True if every node in the tree has exactly 2 outcomes."""
        return all(m == 2 for m in self.outcomes_per_node)

    # -- internal raw access (0-based, for downstream modules) --------------

    def _category_path_raw(self, k: int) -> dict[int, int]:
        """Return (0-based node index → pseudo-response) for category k.

        **Internal.** Used by data_prep.py and cross-tree property tests.
        Only includes nodes on the path (no NA entries).
        """
        row = self.mapping[k, :]
        return {
            r: int(row[r])
            for r in range(self.n_nodes)
            if not np.isnan(row[r])
        }

    def _relevant_categories_raw(self, r: int) -> np.ndarray:
        """Return 0-based category indices that pass through node r.

        **Internal.** Returns a sorted ndarray of ints.
        """
        return np.where(~np.isnan(self.mapping[:, r]))[0]

    # -- display (user-facing, 1-based) -------------------------------------

    def __repr__(self) -> str:
        lines = [
            f"IRTreeSpec(K={self.n_categories}, R={self.n_nodes}, "
            f"latent_dims={self.n_latent_dims})",
            f"  Categories:        K = {self.n_categories}",
            f"  Nodes:             R = {self.n_nodes}",
            f"  Latent dimensions: {self.n_latent_dims}",
            f"  Outcomes per node: {self.outcomes_per_node}",
        ]
        return "\n".join(lines)

    def summary(self) -> str:
        """Human-readable summary of the tree specification.

        All node numbering is **1-based** in the output for consistency
        with the IRTree literature.  Pseudo-response values are displayed
        in the user's convention: if the spec was built with
        ``values_start=1``, the summary shows 1-based values.
        """
        vs = self.values_start
        lines = [
            repr(self),
            "",
            f"Mapping matrix M [{self.n_categories} x {self.n_nodes}]"
            + (f"  (values {vs}-based):" if vs == 1 else ":"),
        ]

        # ---- Column names (via _node_label for consistency) ----
        col_names = [self._node_label(r) for r in range(self.n_nodes)]

        # ---- Row names (via _cat_label for consistency) ----
        row_names = [self._cat_label(k) for k in range(self.n_categories)]

        # ---- Compute column widths ----
        row_width = max(len(r) for r in row_names) + 2
        col_width = max(6, *(len(c) for c in col_names))

        # ---- Header ----
        header = " " * (row_width + 2) + "  ".join(
            f"{c:>{col_width}s}" for c in col_names
        )
        lines.append(header)

        # ---- Data rows (display values shifted by values_start) ----
        for k, label in enumerate(row_names):
            vals = []
            for r in range(self.n_nodes):
                v = self.mapping[k, r]
                if np.isnan(v):
                    vals.append(f"{'NA':>{col_width}s}")
                else:
                    vals.append(f"{int(v) + vs:>{col_width}d}")
            line = f"{label:>{row_width}s}  " + "  ".join(vals)
            lines.append(line)

        # ---- Latent dimension mapping (1-based display) ----
        if self.node_to_latent is not None:
            lines.append("")
            latent_str = ", ".join(
                f"{self._node_label(r)} ← θ{d + 1}"
                for r, d in enumerate(self.node_to_latent)
            )
            lines.append(f"  Latent sharing: {latent_str}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory functions for canonical trees
# ---------------------------------------------------------------------------

def distractor_tree() -> IRTreeSpec:
    """Distractor tree for multiple-choice items.

    An illustrative tree of the kind used for distractor analysis of
    multiple-choice items (e.g., Wright et al., 2008).  Like the other
    factories it is an example structure; any tree can be built with
    ``IRTreeSpec(mapping=...)``.

    References
    ----------
    Wright, S. B., Matlen, B. J., Baym, C. L., Ferrer, E., & Bunge, S. A.
    (2008). Neural correlates of fluid reasoning in children and adults.
    *Frontiers in Human Neuroscience*, 1(8), 1-8.

    4 categories, 3 nodes, nested non-linear tree.

    Structure::

        Node 1 (θ₁: semantic)
        ├─ 0 → Node 2 (θ₂: visual)
        │       ├─ 0 → Cat 1 (unrelated)
        │       └─ 1 → Cat 2 (visual match)
        └─ 1 → Node 3 (θ₃: correctness)
                ├─ 0 → Cat 3 (semantic, wrong)
                └─ 1 → Cat 4 (correct)
    """
    M = np.array([
        [0, 0, np.nan],   # Cat 1: semantic=0, visual=0
        [0, 1, np.nan],   # Cat 2: semantic=0, visual=1
        [1, np.nan, 0],   # Cat 3: semantic=1, correctness=0
        [1, np.nan, 1],   # Cat 4: semantic=1, correctness=1
    ])
    return IRTreeSpec(
        mapping=M,
        category_names=["unrelated", "visual", "semantic", "correct"],
        node_names=["semantic", "visual", "correctness"],
    )


def disengagement_tree(binary: bool = True) -> IRTreeSpec:
    """Disengagement tree (rapid guessing, then ability).

    An illustrative tree for test-taking disengagement: the dichotomous
    version follows the IRTree model of disengagement in Leventhal and
    Pastor (2024); the trichotomous version (partial credit at the ability
    node) follows Mehou et al. (2026).

    References
    ----------
    Leventhal, B. C., & Pastor, D. (2024). An illustration of an IRTree
    model for disengagement. *Educational and Psychological Measurement*,
    84(4), 810-834.

    Mehou, J. M., Leventhal, B. C., Hunsberger, J., & Bao, Y. (2026). An
    IRTree approach to investigating examinee-level predictors of
    disengagement on the PISA. *International Journal of Testing*, 26(2),
    81-103. https://doi.org/10.1080/15305058.2025.2609068

    Parameters
    ----------
    binary : bool
        If True (default), the dichotomous version (K=3, R=2).
        If False, the trichotomous version (K=4, R=2) where the ability
        node has 3 outcomes (incorrect / partial / correct).

    Structure (dichotomous)::

        Node 1 (θ₁: disengagement)
        ├─ 1 → Cat 1 (rapid guess)          [terminal]
        └─ 0 → Node 2 (θ₂: ability)
                ├─ 0 → Cat 2 (incorrect)
                └─ 1 → Cat 3 (correct)

    Structure (trichotomous)::

        Node 1 (θ₁: disengagement)
        ├─ 1 → Cat 1 (rapid guess)          [terminal]
        └─ 0 → Node 2 (θ₂: ability, 3 outcomes)
                ├─ 0 → Cat 2 (incorrect)
                ├─ 1 → Cat 3 (partial)
                └─ 2 → Cat 4 (correct)
    """
    if binary:
        M = np.array([
            [1, np.nan],   # Cat 1: rapid guess
            [0, 0],        # Cat 2: not rapid, incorrect
            [0, 1],        # Cat 3: not rapid, correct
        ])
        return IRTreeSpec(
            mapping=M,
            category_names=["rapid", "incorrect", "correct"],
            node_names=["disengagement", "ability"],
        )
    else:
        M = np.array([
            [1, np.nan],   # Cat 1: rapid guess
            [0, 0],        # Cat 2: incorrect
            [0, 1],        # Cat 3: partial
            [0, 2],        # Cat 4: correct
        ])
        return IRTreeSpec(
            mapping=M,
            category_names=["rapid", "incorrect", "partial", "correct"],
            node_names=["disengagement", "ability"],
        )


def ers_tree() -> IRTreeSpec:
    """Extreme response style (ERS) tree.

    An illustrative response-style tree in the spirit of Böckenholt (2012)
    and Leventhal (2019): a direction decision followed by an extremity
    decision on each branch.

    References
    ----------
    Böckenholt, U. (2012). Modeling multiple response processes in judgment
    and choice. *Psychological Methods*, 17(4), 665-678.

    Leventhal, B. C. (2019). Extreme response style: A simulation study
    comparison of three multidimensional item response models. *Applied
    Psychological Measurement*, 43(4), 322-335.

    4 categories, **3 nodes**, 2 latent dimensions.

    The ERS model postulates two sequential decisions:
    (1) Direction of endorsement (negative vs positive), driven by θ₁.
    (2) Extremity of response (extreme vs non-extreme), driven by θ₂.

    Crucially, the extremity decision occurs on **both branches** of the
    direction node but is driven by the **same** latent variable θ₂.
    This is modelled as two structurally separate nodes (extremity_neg
    and extremity_pos) that share one latent dimension via
    ``node_to_latent=(1, 2, 2)``.

    Structure::

        Node 1 (θ₁: endorsement direction)
        ├─ 0 → Node 2 (θ₂: extremity — negative branch)
        │       ├─ 0 → Cat 2 (negative, non-extreme)
        │       └─ 1 → Cat 1 (negative, extreme)
        └─ 1 → Node 3 (θ₂: extremity — positive branch, same θ)
                ├─ 0 → Cat 3 (positive, non-extreme)
                └─ 1 → Cat 4 (positive, extreme)

    Mapping matrix::

        |              | Y*₁ | Y*₂ | Y*₃ |
        | neg_extreme  |  0  |  1  | NA  |
        | neg_moderate |  0  |  0  | NA  |
        | pos_moderate |  1  | NA  |  0  |
        | pos_extreme  |  1  | NA  |  1  |

    Nodes 2 and 3 share θ₂ but have **separate item parameter arrays**
    (b_extremity_neg and b_extremity_pos), so this model asks whether
    items are equally extreme-response-inducing regardless of endorsement
    direction — a testable claim, not a built-in assumption.

    Relationship to the 2-column representation
    ---------------------------------------------
    A 2-column matrix (no NaN, R=2) is also a valid ERS representation.
    It is exactly this 3-column form under the additional constraint that
    b_extremity_neg = b_extremity_pos (item parameters are the same on
    both branches).  Impose that constraint by passing
    ``Constraints(b_shared=[[2, 3]])`` to ``irtree_fit()`` — this tells
    the sampler that nodes 2 and 3 (1-based: extremity_neg and
    extremity_pos) share a single b array.  The 3-column form used here
    keeps that constraint testable rather than assumed, and makes the
    node/latent-dimension distinction explicit.

    Note on ``n_latent_dims``
    --------------------------
    ``spec.n_nodes == 3`` but ``spec.n_latent_dims == 2``.  The ERS
    tree is the canonical teaching example for ``node_to_latent``: three
    structural decision nodes, two distinct trait dimensions.
    """
    M = np.array([
        [0,  1,      np.nan],   # neg_extreme:  endorse=0, extremity_neg=1
        [0,  0,      np.nan],   # neg_moderate: endorse=0, extremity_neg=0
        [1,  np.nan, 0     ],   # pos_moderate: endorse=1, extremity_pos=0
        [1,  np.nan, 1     ],   # pos_extreme:  endorse=1, extremity_pos=1
    ])
    return IRTreeSpec(
        mapping=M,
        category_names=["neg_extreme", "neg_moderate",
                        "pos_moderate", "pos_extreme"],
        node_names=["endorsement", "extremity_neg", "extremity_pos"],
        node_to_latent=(1, 2, 2),   # extremity_neg & extremity_pos share θ₂
    )



