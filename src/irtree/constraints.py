"""
irtree.constraints
==================
Layer 2b — Item-parameter sharing and cross-node correlation groups.

SHARING GROUPS
--------------
A sharing group forces two or more nodes to use exactly one shared parameter
array.  This reduces the number of estimated parameters and encodes
substantive hypotheses (e.g., "items have the same difficulty at nodes 1
and 2").

    Constraints(b_shared=[[1, 2]])       # nodes 1 & 2 share one b array
    Constraints(a_shared=[[3, 4]])       # nodes 3 & 4 share one a array
    Constraints(b_shared=[[1,2],[3,4]])  # two independent sharing groups

Node indices are 1-based throughout the public API.

CORRELATION GROUPS  (b_corr, a_corr)
--------------------------------------
A correlation group lets the difficulty (b) or discrimination (a) of item i
vary independently across nodes, but draws those values *jointly* from a
multivariate Normal:

    b_i = (b_{i,g1}, ..., b_{i,gG})  ~  N_G(0, Σ_b)

Each sub-list is one correlation group specified as a list of 1-based node
indices, one representative from each b-sharing group:

    Constraints(b_corr=[[1, 3]])          # correlate b-group of node 1
                                          # with b-group of node 3
    Constraints(b_corr=[[1,3],[2,4]])     # two independent corr groups

Restrictions enforced by validate():
  Condition 8:  Every node in a b_corr group must have n_outcomes == 2.
  Condition 9:  No two nodes from the same b-sharing group in one b_corr group.
  Condition 10: Every node in an a_corr group must have has_discrimination=True.
  Condition 11: No two nodes from the same a-sharing group in one a_corr group.

SAMPLER-FACING API
------------------
b_groups_for(R)            → list[list[int]]  0-based group indices
b_node_to_group_for(R)     → list[int]        node r → its b-group index
b_representative_nodes(R)  → list[int]        one rep node per b-group
b_corr_groups_for(R)       → list[list[int]]  corr groups as b-group indices
b_node_to_corr_group_for(R)→ list[int|None]   node r → corr group or None

(Same set for a and c.)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_groups(shared: list[list[int]], R: int) -> list[list[int]]:
    """Convert 1-based sharing declarations into 0-based group lists.

    Shared groups come first (in declaration order), followed by singleton
    groups for the remaining nodes in ascending order.
    """
    groups: list[list[int]] = []
    seen: set = set()
    for grp_1based in shared:
        grp_0 = [n - 1 for n in grp_1based]
        groups.append(grp_0)
        seen.update(grp_0)
    for r in range(R):
        if r not in seen:
            groups.append([r])
    return groups


def _node_to_group(groups: list[list[int]], R: int) -> list[int]:
    """Map each node (0-based) to its group index."""
    n2g = [0] * R
    for g, grp in enumerate(groups):
        for r in grp:
            n2g[r] = g
    return n2g


def _representative_nodes(groups: list[list[int]]) -> list[int]:
    """Return the minimum node index in each group (0-based)."""
    return [min(grp) for grp in groups]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

@dataclass
class Constraints:
    """Declares parameter sharing and cross-node correlation groups.

    Parameters
    ----------
    b_shared : list[list[int]]
        Each sub-list is a group of 1-based node indices that share one b
        (item difficulty) array.
    a_shared : list[list[int]]
        Sharing groups for a (discrimination).  All nodes must have
        has_discrimination=True.
    c_shared : list[list[int]]
        Sharing groups for c (guessing).  All nodes must be probit 3PL
        nodes; a group containing a logit 3PL node is not supported and
        raises NotImplementedError.
    gamma_shared : list[list[int]]
        Sharing groups for gamma (person×item covariate coefficients).
        Each sub-list is a group of 1-based node indices that share ONE
        gamma vector.

        Unlike beta and kappa — whose pooling is *forced* by shared theta
        / shared b (a coefficient regressing on a shared latent quantity
        is otherwise non-identified) — gamma is separately identified at
        every node even when theta and b are both shared, because gamma
        multiplies the independent W tensor and lives in each node's own
        augmented regression. Pooling gamma is therefore a *modeling
        assertion* ("the person×item interaction is a property of a shared
        construct, not of the individual node"), never a correctness
        requirement. So it is opt-in here and independent of node_to_latent
        and b_shared: a user may pool gamma across any node set, whatever
        the theta / b structure.

        Every node in a gamma_shared group MUST declare pi_covariates
        (it cannot be None), and all members must declare the *identical*
        pi_covariates selection. A shared coefficient is one vector whose
        position k means "the effect of W-column c_k"; if two members
        selected different columns, position k would denote two different
        effects and the pooled draw would sum evidence about distinct
        quantities into one number — incoherent, not merely loose. A
        variable either contributes to the whole group or is not part of
        it. (These rules are enforced in
        data_prep.resolve_covariate_indices, since that is where covariate
        selections are known.)
    b_corr : list[list[int]]
        Cross-node b-correlation groups.  Each sub-list contains 1-based
        node indices (one from each sharing group to be correlated).
        All nodes must have n_outcomes==2.
    a_corr : list[list[int]]
        Cross-node a-correlation groups.  All nodes must have
        has_discrimination=True.
    theta_corr : list[list[int]]
        Latent-dimension correlation groups. Each sub-list contains 1-based
        LATENT DIMENSION indices (not node indices) that form one estimated
        correlation block; dimensions outside every block stay independent.
        This is the theta analogue of b_corr / a_corr, and it is what makes
        it possible to estimate a correlation among a *subset* of latent
        dimensions under LKJ / InverseWishart (the theta.correlation prior,
        LKJ(1) by default, otherwise spans all D dimensions at once). Empty → the prior applies
        to all D dimensions jointly.
        Blocks must be disjoint (each dimension in at most one block) and use
        valid 1-based dimension indices in 1..D. A singleton block is not a
        correlation and is rejected.
    """

    b_shared: list[list[int]] = field(default_factory=list)
    a_shared: list[list[int]] = field(default_factory=list)
    c_shared: list[list[int]] = field(default_factory=list)
    gamma_shared: list[list[int]] = field(default_factory=list)
    b_corr:   list[list[int]] = field(default_factory=list)
    a_corr:   list[list[int]] = field(default_factory=list)
    theta_corr: list[list[int]] = field(default_factory=list)

    # Cache — populated by validate() or lazily by the group methods
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        parts = []
        if self.b_shared:
            parts.append(f"b_shared={self.b_shared}")
        if self.a_shared:
            parts.append(f"a_shared={self.a_shared}")
        if self.c_shared:
            parts.append(f"c_shared={self.c_shared}")
        if self.gamma_shared:
            parts.append(f"gamma_shared={self.gamma_shared}")
        if self.b_corr:
            parts.append(f"b_corr={self.b_corr}")
        if self.a_corr:
            parts.append(f"a_corr={self.a_corr}")
        if self.theta_corr:
            parts.append(f"theta_corr={self.theta_corr}")
        if not parts:
            return "Constraints(no sharing)"
        return "Constraints(" + ", ".join(parts) + ")"

    # ------------------------------------------------------------------
    # Flags
    # ------------------------------------------------------------------

    @property
    def any_sharing(self) -> bool:
        return bool(self.b_shared or self.a_shared or self.c_shared
                    or self.gamma_shared)

    @property
    def has_b_sharing(self) -> bool:
        return bool(self.b_shared)

    @property
    def has_a_sharing(self) -> bool:
        return bool(self.a_shared)

    @property
    def has_c_sharing(self) -> bool:
        return bool(self.c_shared)

    @property
    def has_gamma_sharing(self) -> bool:
        return bool(self.gamma_shared)

    @property
    def has_b_corr(self) -> bool:
        return bool(self.b_corr)

    @property
    def has_a_corr(self) -> bool:
        return bool(self.a_corr)

    @property
    def has_theta_corr(self) -> bool:
        return bool(self.theta_corr)

    # ------------------------------------------------------------------
    # Sharing group methods (b, a, c)
    # ------------------------------------------------------------------

    def b_groups_for(self, R: int) -> list[list[int]]:
        """0-based b-sharing groups for an R-node tree."""
        key = ("b_groups", R)
        if key not in self._cache:
            self._cache[key] = _build_groups(self.b_shared, R)
        return self._cache[key]

    def b_node_to_group_for(self, R: int) -> list[int]:
        """Map node r (0-based) → its b-group index."""
        key = ("b_n2g", R)
        if key not in self._cache:
            self._cache[key] = _node_to_group(self.b_groups_for(R), R)
        return self._cache[key]

    def b_representative_nodes(self, R: int) -> list[int]:
        """Min node index (0-based) per b-group."""
        key = ("b_reps", R)
        if key not in self._cache:
            self._cache[key] = _representative_nodes(self.b_groups_for(R))
        return self._cache[key]

    def a_groups_for(self, R: int) -> list[list[int]]:
        key = ("a_groups", R)
        if key not in self._cache:
            self._cache[key] = _build_groups(self.a_shared, R)
        return self._cache[key]

    def a_node_to_group_for(self, R: int) -> list[int]:
        key = ("a_n2g", R)
        if key not in self._cache:
            self._cache[key] = _node_to_group(self.a_groups_for(R), R)
        return self._cache[key]

    def a_representative_nodes(self, R: int) -> list[int]:
        key = ("a_reps", R)
        if key not in self._cache:
            self._cache[key] = _representative_nodes(self.a_groups_for(R))
        return self._cache[key]

    def c_groups_for(self, R: int) -> list[list[int]]:
        key = ("c_groups", R)
        if key not in self._cache:
            self._cache[key] = _build_groups(self.c_shared, R)
        return self._cache[key]

    def c_node_to_group_for(self, R: int) -> list[int]:
        key = ("c_n2g", R)
        if key not in self._cache:
            self._cache[key] = _node_to_group(self.c_groups_for(R), R)
        return self._cache[key]

    def c_representative_nodes(self, R: int) -> list[int]:
        key = ("c_reps", R)
        if key not in self._cache:
            self._cache[key] = _representative_nodes(self.c_groups_for(R))
        return self._cache[key]

    def gamma_groups_for(self, R: int) -> list[list[int]]:
        """0-based gamma-sharing groups for an R-node tree.

        Shared groups first (declaration order), then singletons for the
        remaining nodes. When ``gamma_shared`` is empty this is the
        all-singletons partition ``[[0], [1], ..., [R-1]]`` — i.e. the
        per-node default, so the pooling machinery degrades cleanly to
        one gamma per node.
        """
        key = ("gamma_groups", R)
        if key not in self._cache:
            self._cache[key] = _build_groups(self.gamma_shared, R)
        return self._cache[key]

    def gamma_node_to_group_for(self, R: int) -> list[int]:
        """Map node r (0-based) → its gamma-group index."""
        key = ("gamma_n2g", R)
        if key not in self._cache:
            self._cache[key] = _node_to_group(self.gamma_groups_for(R), R)
        return self._cache[key]

    def gamma_representative_nodes(self, R: int) -> list[int]:
        """Min node index (0-based) per gamma-group."""
        key = ("gamma_reps", R)
        if key not in self._cache:
            self._cache[key] = _representative_nodes(self.gamma_groups_for(R))
        return self._cache[key]

    # ------------------------------------------------------------------
    # Correlation group methods (b_corr, a_corr)
    # ------------------------------------------------------------------

    def b_corr_groups_for(self, R: int) -> list[list[int]]:
        """Return b-correlation groups as lists of b-GROUP indices (0-based).

        Each sub-list contains the group indices of the sharing groups to
        be correlated.  Empty list when no b_corr was declared.
        """
        key = ("b_corr_groups", R)
        if key not in self._cache:
            n2g = self.b_node_to_group_for(R)
            result = []
            for grp_nodes in self.b_corr:
                result.append([n2g[n - 1] for n in grp_nodes])
            self._cache[key] = result
        return self._cache[key]

    def b_node_to_corr_group_for(self, R: int) -> list[int | None]:
        """Map node r (0-based) → correlation group index, or None."""
        key = ("b_node_corr_g", R)
        if key not in self._cache:
            n2g   = self.b_node_to_group_for(R)
            corr_groups = self.b_corr_groups_for(R)
            result: list[int | None] = [None] * R
            for cg_idx, cg in enumerate(corr_groups):
                cg_set = set(cg)
                for r in range(R):
                    if n2g[r] in cg_set:
                        result[r] = cg_idx
            self._cache[key] = result
        return self._cache[key]

    def a_corr_groups_for(self, R: int) -> list[list[int]]:
        """Return a-correlation groups as lists of a-GROUP indices (0-based)."""
        key = ("a_corr_groups", R)
        if key not in self._cache:
            n2g = self.a_node_to_group_for(R)
            result = []
            for grp_nodes in self.a_corr:
                result.append([n2g[n - 1] for n in grp_nodes])
            self._cache[key] = result
        return self._cache[key]

    def a_node_to_corr_group_for(self, R: int) -> list[int | None]:
        """Map node r (0-based) → a-correlation group index, or None."""
        key = ("a_node_corr_g", R)
        if key not in self._cache:
            n2g   = self.a_node_to_group_for(R)
            corr_groups = self.a_corr_groups_for(R)
            result: list[int | None] = [None] * R
            for cg_idx, cg in enumerate(corr_groups):
                cg_set = set(cg)
                for r in range(R):
                    if n2g[r] in cg_set:
                        result[r] = cg_idx
            self._cache[key] = result
        return self._cache[key]

    def theta_corr_groups_for(self, D: int) -> list[list[int]]:
        """Return latent-dimension correlation blocks as 0-based dim indices.

        Each sub-list is the 0-based latent-dimension indices forming one
        estimated correlation block. Unlike b_corr/a_corr there is NO
        node→group indirection — theta correlation is over latent dimensions
        directly, so the declared 1-based dim indices simply shift to 0-based.
        Empty list when no theta_corr was declared (→ caller falls back to
        the all-D joint correlation).
        """
        key = ("theta_corr_groups", D)
        if key not in self._cache:
            self._cache[key] = [[d - 1 for d in blk] for blk in self.theta_corr]
        return self._cache[key]

    def theta_dim_to_corr_group_for(self, D: int) -> list[int | None]:
        """Map latent dimension d (0-based) → its theta_corr block index, or None."""
        key = ("theta_dim_corr_g", D)
        if key not in self._cache:
            result: list[int | None] = [None] * D
            for cg_idx, blk in enumerate(self.theta_corr_groups_for(D)):
                for d in blk:
                    result[d] = cg_idx
            self._cache[key] = result
        return self._cache[key]

    # ------------------------------------------------------------------
    # validate()
    # ------------------------------------------------------------------

    def validate(self, spec, node_models: list) -> None:
        """Validate all sharing and correlation declarations against the tree.

        Raises ValueError for any violation, and NotImplementedError for a
        c_shared group containing a logit 3PL node.  On success, populates the
        internal group cache so subsequent calls to the group methods are free.

        Parameters
        ----------
        spec : IRTreeSpec-like
            Must have .n_nodes attribute.
        node_models : list
            R NodeModel-like objects with .n_outcomes, .has_discrimination,
            .has_guessing attributes.
        """
        R = spec.n_nodes
        # Clear cache so fresh group computations pick up spec
        self._cache.clear()

        # Validate sharing groups (conditions 1–7)
        self._validate_sharing(R, node_models, "b_shared", self.b_shared,
                               _check_b_compatible)
        self._validate_sharing(R, node_models, "a_shared", self.a_shared,
                               _check_a_compatible)
        self._validate_sharing(R, node_models, "c_shared", self.c_shared,
                               _check_c_compatible)
        self._validate_sharing(R, node_models, "gamma_shared",
                               self.gamma_shared, _check_gamma_compatible)

        # Validate b_corr groups (conditions 8–9)
        if self.b_corr:
            b_n2g = self.b_node_to_group_for(R)
            for ci, grp in enumerate(self.b_corr):
                # Basic range / size checks (same as sharing)
                self._check_group_range(grp, R, f"b_corr[{ci}]")
                self._check_group_min_size(grp, 2, f"b_corr[{ci}]")
                # Condition 8: all nodes must have scalar b (n_outcomes == 2)
                for n in grp:
                    nm = node_models[n - 1]
                    is_mixed = getattr(nm, "is_mixed", False)
                    if nm.n_outcomes != 2 or is_mixed:
                        raise ValueError(
                            f"b_corr group {ci+1} (nodes {grp}): node {n} has "
                            f"n_outcomes={nm.n_outcomes} (mixed={is_mixed}).  "
                            f"b.correlation is only supported for binary nodes "
                            f"(n_outcomes==2, not mixed-format).  "
                            f"GRMNode(n_outcomes=2) is acceptable."
                        )
                # Condition 9: no two nodes from the same b-sharing group
                g_indices = [b_n2g[n - 1] for n in grp]
                if len(g_indices) != len(set(g_indices)):
                    dupes = [n for n in grp
                             if g_indices.count(b_n2g[n - 1]) > 1]
                    raise ValueError(
                        f"b_corr group {ci+1} (nodes {grp}): nodes {dupes} "
                        f"belong to the same b-sharing group.  Nodes in the "
                        f"same sharing group share one b array and cannot be "
                        f"in the same correlation group.  Specify one "
                        f"representative node per sharing group."
                    )

        # Validate a_corr groups (conditions 10–11)
        if self.a_corr:
            a_n2g = self.a_node_to_group_for(R)
            for ci, grp in enumerate(self.a_corr):
                self._check_group_range(grp, R, f"a_corr[{ci}]")
                self._check_group_min_size(grp, 2, f"a_corr[{ci}]")
                # Condition 10: all nodes must have has_discrimination=True
                for n in grp:
                    nm = node_models[n - 1]
                    if not nm.has_discrimination:
                        raise ValueError(
                            f"a_corr group {ci+1} (nodes {grp}): node {n} has "
                            f"has_discrimination=False.  a.correlation requires "
                            f"2PL or 3PL nodes (has_discrimination=True) in every "
                            f"position of the correlation group."
                        )
                    if getattr(nm, "common_discrimination", False):
                        raise ValueError(
                            f"a_corr group {ci+1} (nodes {grp}): node {n} has "
                            f"common_discrimination=True.  a.correlation "
                            f"correlates the slopes of the same item across "
                            f"nodes, which needs one slope per item."
                        )
                # Condition 11: no two nodes from the same a-sharing group
                g_indices = [a_n2g[n - 1] for n in grp]
                if len(g_indices) != len(set(g_indices)):
                    dupes = [n for n in grp
                             if g_indices.count(a_n2g[n - 1]) > 1]
                    raise ValueError(
                        f"a_corr group {ci+1} (nodes {grp}): nodes {dupes} "
                        f"belong to the same a-sharing group.  Specify one "
                        f"representative node per sharing group."
                    )

        # Validate theta_corr blocks (latent-dimension correlation subsets)
        if self.theta_corr:
            D = getattr(spec, "n_latent_dims", R)
            seen_dims: dict[int, int] = {}   # dim (1-based) → block index
            for ci, blk in enumerate(self.theta_corr):
                # Range: 1-based dimension indices within 1..D
                for d in blk:
                    if d < 1 or d > D:
                        raise ValueError(
                            f"theta_corr block {ci+1} ({blk}): dimension {d} is "
                            f"out of range for a {D}-latent-dimension model "
                            f"(valid: 1–{D}). Note theta_corr indexes LATENT "
                            f"DIMENSIONS, not nodes."
                        )
                # Min size: a singleton is not a correlation
                if len(blk) < 2:
                    raise ValueError(
                        f"theta_corr block {ci+1} ({blk}) has fewer than 2 "
                        f"dimensions. A correlation block needs at least 2; a "
                        f"single dimension has nothing to correlate with."
                    )
                # No repeats within a block
                if len(blk) != len(set(blk)):
                    raise ValueError(
                        f"theta_corr block {ci+1} ({blk}) lists a dimension "
                        f"more than once."
                    )
                # Disjointness: each dimension in at most one block
                for d in blk:
                    if d in seen_dims:
                        raise ValueError(
                            f"theta_corr: dimension {d} appears in block "
                            f"{seen_dims[d]+1} and block {ci+1}. Blocks must be "
                            f"disjoint (each latent dimension in at most one "
                            f"correlated block)."
                        )
                    seen_dims[d] = ci

    # ------------------------------------------------------------------
    # summary()
    # ------------------------------------------------------------------

    def summary(self, R: int | None = None) -> None:
        """Print a human-readable summary of all sharing and correlation groups."""
        print("Constraints summary")
        print("─" * 50)
        if not self.any_sharing and not self.b_corr and not self.a_corr and not self.theta_corr:
            print("  No sharing or correlation declared (all independent).")
            return
        for label, shared in (
            ("b_shared", self.b_shared),
            ("a_shared", self.a_shared),
            ("c_shared", self.c_shared),
        ):
            if shared:
                for grp in shared:
                    print(f"  {label}:  nodes {grp} share one array")
        for label, corr in (("b_corr", self.b_corr), ("a_corr", self.a_corr)):
            if corr:
                for grp in corr:
                    print(f"  {label}:  nodes {grp} drawn jointly (correlated)")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_sharing(self, R, node_models, attr_name, shared_list,
                           compat_fn):
        """Validate a single sharing list (b_shared / a_shared / c_shared)."""
        all_nodes_seen: dict[int, int] = {}  # node_0 → group index
        for gi, grp in enumerate(shared_list):
            self._check_group_range(grp, R, f"{attr_name}[{gi}]")
            self._check_group_min_size(grp, 2, f"{attr_name}[{gi}]")
            for n in grp:
                n0 = n - 1
                if n0 in all_nodes_seen:
                    raise ValueError(
                        f"{attr_name}: node {n} appears in both group "
                        f"{all_nodes_seen[n0]+1} and group {gi+1}.  "
                        f"Each node can belong to at most one sharing group."
                    )
                all_nodes_seen[n0] = gi
            compat_fn(grp, node_models, attr_name, gi)

    @staticmethod
    def _check_group_range(grp, R, label):
        for n in grp:
            if n < 1 or n > R:
                raise ValueError(
                    f"{label}: node index {n} is out of range for a "
                    f"{R}-node tree (valid: 1–{R})."
                )

    @staticmethod
    def _check_group_min_size(grp, min_size, label):
        if len(grp) < min_size:
            raise ValueError(
                f"{label} must contain at least {min_size} nodes; "
                f"got {grp}."
            )


# ---------------------------------------------------------------------------
# Per-parameter compatibility checkers
# ---------------------------------------------------------------------------

def _check_b_compatible(grp, node_models, attr_name, gi):
    """b-sharing: all nodes must have same n_outcomes and pseudo_item_outcomes."""
    nms = [node_models[n - 1] for n in grp]
    # n_outcomes must be identical
    outcomes = [nm.n_outcomes for nm in nms]
    if len(set(outcomes)) > 1:
        raise ValueError(
            f"{attr_name}[{gi}] (nodes {grp}): nodes have differing "
            f"n_outcomes {outcomes}.  Nodes must have the same n_outcomes "
            f"to share item difficulty parameters."
        )
    # Mixed-node check: if any node is mixed, all must be mixed with the
    # same pseudo_item_outcomes vector.
    import numpy as np
    mixed_flags = [getattr(nm, "is_mixed", False) for nm in nms]
    has_mixed   = any(mixed_flags)
    all_mixed   = all(mixed_flags)
    if has_mixed:
        if not all_mixed:
            non_mixed = [grp[i] for i, f in enumerate(mixed_flags) if not f]
            raise ValueError(
                f"{attr_name}[{gi}] (nodes {grp}): mixed-format nodes "
                f"cannot share b with homogeneous nodes.  "
                f"Non-mixed nodes in group: {non_mixed}."
            )
        # All mixed — check pseudo_item_outcomes compatibility
        pio_list = [nm.pseudo_item_outcomes for nm in nms]
        ref = pio_list[0]
        for i, pio in enumerate(pio_list[1:], 1):
            if not np.array_equal(pio, ref):
                raise ValueError(
                    f"{attr_name}[{gi}] (nodes {grp}): mixed-format nodes "
                    f"have differing pseudo_item_outcomes arrays.  "
                    f"Node {grp[0]}: {ref.tolist()},  "
                    f"Node {grp[i]}: {pio.tolist()}.  "
                    f"Both must have identical per-item outcome counts to "
                    f"share a b array."
                )


def _check_a_compatible(grp, node_models, attr_name, gi):
    """a-sharing: all nodes must have has_discrimination=True and none may
    declare common_discrimination (one slope for the node), which the pooled
    per-item update of a shared array does not support."""
    for n in grp:
        nm = node_models[n - 1]
        if not nm.has_discrimination:
            raise ValueError(
                f"{attr_name}[{gi}] (nodes {grp}): node {n} has "
                f"has_discrimination=False.  Only 2PL / 3PL nodes "
                f"(has_discrimination=True) can share discrimination parameters."
            )
        if getattr(nm, "common_discrimination", False):
            raise NotImplementedError(
                f"{attr_name}[{gi}] (nodes {grp}): node {n} has "
                f"common_discrimination=True.  A single node-wide slope cannot be "
                f"part of an a-sharing group; drop one of the two."
            )


def _check_c_compatible(grp, node_models, attr_name, gi):
    """c-sharing: all nodes must be probit 3PL nodes.

    Shared guessing across a logit 3PL node is not supported: the pooled c
    draw is the Beta Gibbs step given the probit 3PL guessing indicators U,
    which a logit node does not have.
    """
    for n in grp:
        nm = node_models[n - 1]
        if not nm.has_guessing:
            raise ValueError(
                f"{attr_name}[{gi}] (nodes {grp}): node {n} has "
                f"has_guessing=False.  Only 3PL nodes (has_guessing=True) "
                f"can share guessing parameters."
            )
        if not nm.augmentable:
            raise NotImplementedError(
                f"{attr_name}[{gi}] (nodes {grp}): node {n} is a logit 3PL node.  "
                f"Shared guessing is supported between probit 3PL nodes only "
                f"(the pooled c draw uses the probit guessing indicators U)."
            )


def _check_gamma_compatible(grp, node_models, attr_name, gi):
    """gamma-sharing: every node in the group must DECLARE pi_covariates.

    A node that declares no person×item covariate contributes nothing to a
    shared gamma vector, so it cannot be a member of a gamma-sharing group.
    (This is the structural half of the rule. The other half — that all
    members must select the *identical* W columns — is checked in
    data_prep.resolve_covariate_indices, where the name/index selections
    have been resolved to concrete column arrays.)
    """
    for n in grp:
        nm = node_models[n - 1]
        if not getattr(nm, "has_pi_covariates", False):
            raise ValueError(
                f"{attr_name}[{gi}] (nodes {grp}): node {n} declares no "
                f"pi_covariates.  Every node in a gamma-sharing group must "
                f"declare person×item covariates (pi_covariates cannot be "
                f"None) — a node contributing nothing to the interaction "
                f"cannot share its coefficient.  Either add pi_covariates to "
                f"node {n} or remove it from the gamma_shared group."
            )


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def share_nodes(*node_groups: list[int]) -> Constraints:
    """Convenience constructor: share b across node groups.

    Example::

        share_nodes([1, 2], [3, 4])
        # equivalent to Constraints(b_shared=[[1, 2], [3, 4]])
    """
    return Constraints(b_shared=list(node_groups))


# Sentinel: no constraints at all (all parameters independent, no correlation)
NO_CONSTRAINTS = Constraints()
