"""
irtree.data_prep
================
Layer 2 — Data preparation: raw response matrix → PreparedData.

PreparedData
------------
A NamedTuple that holds everything the sampler needs downstream.
The response-related fields are built by ``expand_data`` from the raw
response matrix Y and the tree specification.  The covariate fields are
built from the optional X, Q, W matrices.

**Response fields** (always present)

Y_star : list[ndarray [P, I]]
    Pseudo-response values at each node.  ``Y_star[r][p, i]`` is the
    value from the mapping matrix for person p's response to item i at
    node r.  ``np.nan`` when node r is not on the path to that response
    (structurally irrelevant) or when the response is missing.

T : list[ndarray [P, I], dtype bool]
    Combined structural-relevance + missingness gate.
    ``T[r][p, i] = True`` iff node r is on the path to person p's
    response on item i AND the response is present (not missing).
    This is the single gate that controls all downstream computation:
    likelihoods, Gibbs samplers, and chain storage all condition on T.

row_idx, col_idx : list[ndarray]
    Precomputed ``np.where(T[r])`` for each node.  Used by
    ``gibbs_samplers.py`` for pair enumeration.

observed : ndarray [P, I], dtype bool
    ``True`` when person p's response on item i is present.

n_missing : int
    Total count of missing (p, i) cells across the full P×I matrix.

n_persons, n_items, n_nodes : int

**Covariate fields** (None / 0 when the corresponding matrix is not supplied)

X : ndarray [P, K_x] or None
    Person-level covariate matrix.  Missing CELLS are filled with 0.0 (the
    fill value prevents NaN propagation and is overwritten by the sampler's
    imputation); observed cells keep their values, including the observed
    features of a person whose row is only partially missing.

Q : ndarray [I, K_q] or None
    Item-level covariate matrix.  Same fill convention.

W : ndarray [P, I, K_w] or None
    Person-by-item covariate tensor.

X_observed : ndarray [P], dtype bool, or None
    ``True`` when all K_x covariate values for person p are present.
    Row-wise: a person is fully observed iff every covariate is present.

Q_observed : ndarray [I], dtype bool, or None
    ``True`` when all K_q covariate values for item i are present.

W_observed : ndarray [P, I], dtype bool, or None
    ``True`` when all K_w covariate values for cell (p, i) are present.

X_feature_observed, Q_feature_observed, W_feature_observed : ndarray bool or None
    Per-value masks, same shape as X, Q, W: ``True`` where that individual
    covariate value is present.  The sampler imputes exactly the False cells.

X_names, Q_names, W_names : tuple[str] or None
    Covariate column names, used for node covariate selection by name and
    for chain column names (beta_{group}_{name}); None when unnamed.

n_missing_X, n_missing_Q, n_missing_W : int
    Missing-person / item / cell counts within the respective covariate
    matrices.  0 when the corresponding matrix is None.

**Entity id fields** (None when not supplied, e.g. the raw-array path)

person_ids : tuple or None
    The researcher's own person identifiers, length P, in the same
    (canonical) order as the rows of Y.  ``person_ids[p]`` labels row p.
    Populated by ``ingest.load_data``; None when Y was passed as a bare
    array.  Used only to relabel per-person parameters in output; the
    sampler itself is id-agnostic and works purely positionally.

item_ids : tuple or None
    The researcher's own item identifiers, length I, aligned to the
    columns of Y.  Same role as ``person_ids``, one axis over.

Missing covariate convention
----------------------------
Missing covariate values are treated as parameters and sampled at each
iteration from their full conditional (see ``gibbs_samplers.gibbs_X_missing``).
The ``X`` array stored here is the *starting point* for the sampler's working
copy: its missing cells are seeded to 0.0 (see the field note above) and the
sampler overwrites them at every iteration, so every person contributes to the
β update, and to the likelihood, with that iteration's imputed values (full
imputation; the same holds for Q and W).  Imputation is Gibbs from the probit
nodes' latent regression, or MH when a logit/GPCM node uses the covariate.  Only missing cells are
seeded; observed values in a partially-missing row are kept. The ``X_observed`` mask permanently
records which rows were originally fully observed, and the per-cell
``X_feature_observed`` mask records which individual values were, so the
sampler knows exactly which cells to impute.

Design note: covariate arrays live in PreparedData, NOT in NodeModel.
``NodeModel`` instances carry only the covariate SELECTION
(``person_covariates`` / ``item_covariates`` / ``pi_covariates``: None,
True or a list of names/indices, with derived ``has_*`` flags), which
``resolve_covariate_indices`` turns into column indices once per run.  Covariate data travels through the same ``data`` argument channel
as Y_star and T, which keeps it compatible with ``ProcessPoolExecutor``
pickling.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

# ===========================================================================
# PreparedData
# ===========================================================================

class PreparedData(NamedTuple):
    """All data the sampler needs, pre-processed once and reused each iteration.

    See module docstring for field descriptions.
    """
    # --- response data ---
    Y_star:     list        # list[ndarray [P, I] float64]  pseudo-responses
    T:          list        # list[ndarray [P, I] bool]     relevance + obs gate
    row_idx:    list        # list[ndarray int]             np.where(T[r])[0]
    col_idx:    list        # list[ndarray int]             np.where(T[r])[1]
    observed:   np.ndarray  # [P, I] bool
    n_missing:  int
    n_persons:  int
    n_items:    int
    n_nodes:    int
    # --- person covariates ---
    X:                  np.ndarray | None   # [P, K_x] float64 or None
    X_observed:         np.ndarray | None   # [P] bool or None  — row is fully observed
    X_feature_observed: np.ndarray | None   # [P, K_x] bool or None — per-cell mask
    n_missing_X:        int                    # 0 when X is None
    # --- item covariates ---
    Q:                  np.ndarray | None   # [I, K_q] float64 or None
    Q_observed:         np.ndarray | None   # [I] bool or None
    Q_feature_observed: np.ndarray | None   # [I, K_q] bool or None — per-cell mask
    n_missing_Q:        int
    # --- person×item covariates ---
    W:                  np.ndarray | None   # [P, I, K_w] float64 or None
    W_observed:         np.ndarray | None   # [P, I] bool or None
    W_feature_observed: np.ndarray | None   # [P, I, K_w] bool or None — per-cell mask
    n_missing_W:        int
    # --- covariate column names (optional; enable name-based node selection) ---
    X_names: tuple[str, ...] | None   # length K_x, or None if X is None / unnamed
    Q_names: tuple[str, ...] | None   # length K_q, or None if Q is None / unnamed
    W_names: tuple[str, ...] | None   # length K_w, or None if W is None / unnamed
    # --- entity id labels (optional; enable id-based result labelling) ---
    # Trailing defaults so existing positional/keyword construction stays valid.
    person_ids: tuple | None = None   # length P, canonical order, or None
    item_ids:   tuple | None = None   # length I, canonical order, or None


# ===========================================================================
# Internal helpers
# ===========================================================================

def _build_observed(Y: np.ndarray, missing_codes: tuple) -> np.ndarray:
    """Build [P, I] boolean observed mask from Y and missing_codes."""
    P, I = Y.shape
    is_nan = np.zeros((P, I), dtype=bool)
    try:
        is_nan = np.isnan(Y.astype(float))
    except (ValueError, TypeError):
        pass

    is_sentinel = np.zeros((P, I), dtype=bool)
    for code in missing_codes:
        is_sentinel |= (Y == code)

    return ~(is_nan | is_sentinel)


def _print_missing_message(
    n_missing: int,
    P: int,
    I: int,
    observed: np.ndarray,
    label: str = "response",
) -> None:
    """Print user-facing missing-data summary."""
    pct = 100.0 * n_missing / (P * I)
    print(
        f"Missing data detected ({label}): {n_missing} cells "
        f"({pct:.1f}% of {P}×{I} matrix). "
        f"Missing cells excluded from likelihood "
        f"(MCAR/MAR valid; MNAR produces biased estimates)."
    )
    # Per-item detail for items with > 10% missing
    n_missing_per_item = np.sum(~observed, axis=0)   # [I]
    threshold = 0.1 * P
    bad_items = np.where(n_missing_per_item > threshold)[0]
    if len(bad_items) > 0:
        details = ", ".join(
            f"item {i + 1} ({100.0 * n_missing_per_item[i] / P:.1f}%)"
            for i in bad_items
        )
        print(f"  Items exceeding 10% missing: {details}.")


def _process_covariate(
    cov: np.ndarray,
    expected_leading: tuple[int, ...],
    axis_names: tuple[str, ...],
    missing_codes: tuple,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate and process a covariate array of any leading shape.

    Returns
    -------
    cov_out : ndarray [..., K] float64
        Missing cells filled with 0.0; observed cells (including the
        observed features of a partially-missing unit) keep their values.
    observed_out : ndarray [...] bool
        True iff ALL K features present for that unit.
    feature_observed_out : ndarray [..., K] bool
        Per-cell mask: True iff that specific feature value is present.
        Observed units have all True; partially-missing units have a mix.
    n_missing_out : int
        Number of units with at least one missing feature value.
    """
    n_leading = len(expected_leading)
    cov_f = np.asarray(cov, dtype=float)
    if cov_f.ndim == n_leading:
        cov_f = cov_f[..., np.newaxis]
    for dim, (got, want, name) in enumerate(
        zip(cov_f.shape, expected_leading, axis_names, strict=False)
    ):
        if got != want:
            raise ValueError(
                f"{label} has shape {cov_f.shape} but expected dimension "
                f"{dim} ({name}) to be {want}."
            )
    K = cov_f.shape[-1]
    is_nan = np.isnan(cov_f)
    is_sentinel = np.zeros_like(is_nan)
    for code in missing_codes:
        try:
            is_sentinel |= (cov_f == float(code))
        except (ValueError, TypeError):
            pass
    is_missing_cell = is_nan | is_sentinel            # [..., K] per-feature mask
    feature_observed_out = ~is_missing_cell           # [..., K] — NEW
    observed_out = ~np.any(is_missing_cell, axis=-1)  # [...] row-level mask
    all_missing_col = np.all(is_missing_cell.reshape(-1, K), axis=0)
    if np.any(all_missing_col):
        bad = np.where(all_missing_col)[0].tolist()
        axis_desc = "/".join(axis_names)
        raise ValueError(
            f"{label}: feature column(s) {bad} are entirely missing "
            f"(no valid value across all {axis_desc}). "
            f"The model cannot estimate a coefficient for a covariate "
            f"with no observed values."
        )
    cov_out = cov_f.copy()
    # Fill ONLY the missing cells.  The observed values of a partially-missing
    # unit are data: the samplers hold them fixed (via the per-cell
    # ``feature_observed_out`` mask) while drawing the missing ones, so they
    # must survive here.  (Zeroing the whole row would make those zeros be
    # read as observed covariate values.)
    cov_out[is_missing_cell] = 0.0
    n_missing_out = int(np.sum(~observed_out))
    return cov_out, observed_out, feature_observed_out, n_missing_out


def _validate_covariate_names(
    names: Sequence[str] | None,
    n_cols: int,
    label: str,
) -> tuple[str, ...] | None:
    """Validate an optional column-name sequence against a covariate matrix.

    Returns a tuple of names, or None if ``names`` is None.

    Raises
    ------
    ValueError
        If the length doesn't match the matrix's column count, or if any
        name is duplicated (duplicate names make name-based node selection
        in ``resolve_covariate_indices`` ambiguous).
    """
    if names is None:
        return None
    names = tuple(names)
    if len(names) != n_cols:
        raise ValueError(
            f"{label}_names has length {len(names)} but {label} has "
            f"{n_cols} column(s); they must match."
        )
    seen = set()
    dupes = sorted({n for n in names if n in seen or seen.add(n)})
    if dupes:
        raise ValueError(
            f"{label}_names contains duplicate name(s): {dupes}. "
            f"Column names must be unique so node models can select "
            f"covariates unambiguously by name."
        )
    return names


def _print_covariate_missing_message(
    n_missing: int,
    N: int,
    label: str,
) -> None:
    """Print user-facing missing-covariate summary."""
    pct = 100.0 * n_missing / N
    print(
        f"Missing covariate data ({label}): {n_missing} of {N} "
        f"({pct:.1f}%) have at least one missing covariate value. "
        f"Missing covariate values will be sampled as parameters "
        f"(Normal prior) at each MCMC iteration."
    )


# ===========================================================================
# Public API
# ===========================================================================

def expand_data(
    Y,
    spec,
    categories_start: int = 0,
    missing_codes: tuple = (99,),
    X=None,
    Q=None,
    W=None,
    X_missing_codes: tuple = (99,),
    Q_missing_codes: tuple = (99,),
    W_missing_codes: tuple = (99,),
    X_names: Sequence[str] | None = None,
    Q_names: Sequence[str] | None = None,
    W_names: Sequence[str] | None = None,
    person_ids: Sequence | None = None,
    item_ids:   Sequence | None = None,
) -> PreparedData:
    """Expand a raw response matrix into a PreparedData object.

    Parameters
    ----------
    Y : array-like [P, I]
        Raw response matrix.  Integer or float.  Missing values are
        represented by ``np.nan`` or any value in ``missing_codes``.
    spec : IRTreeSpec
        Tree specification from ``tree_spec.py``.
    categories_start : int
        ``1`` if category labels are 1-based (most common in practice),
        ``0`` if 0-based.  Only 0 and 1 are supported.
    missing_codes : tuple of int/float
        Additional sentinels treated as missing.  Default ``(99,)``.
        Pass an empty tuple to detect NaN only.

    X : array-like [P, K_x] or None
        Person-level covariate matrix.  Optional.
    Q : array-like [I, K_q] or None
        Item-level covariate matrix.  Optional.
    W : array-like [P, I, K_w] or None
        Person-by-item covariate tensor.  Optional.
    X_missing_codes : tuple
        Sentinels for missing values in X.  Default ``(99,)``.
    Q_missing_codes : tuple
        Sentinels for missing values in Q.
    W_missing_codes : tuple
        Sentinels for missing values in W.
    X_names : sequence of str, or None
        Column names for X, length K_x. Optional. When given, node models
        may select person covariates by name (``person_covariates=["ses"]``)
        instead of by integer column index. Stored on ``PreparedData.X_names``
        for use by ``resolve_covariate_indices`` and for chain-column naming.
    Q_names : sequence of str, or None
        Column names for Q, length K_q. Same purpose as ``X_names``.
    W_names : sequence of str, or None
        Column names for W, length K_w. Same purpose as ``X_names``.
    person_ids : sequence, or None
        The researcher's person identifiers, length P, aligned to the rows
        of Y. Stored on ``PreparedData.person_ids`` so results can be
        relabelled by id at display time. Normally supplied by
        ``ingest.load_data``; None on the raw-array path (positional labels).
    item_ids : sequence, or None
        The researcher's item identifiers, length I, aligned to the columns
        of Y. Same purpose as ``person_ids``, one axis over.

    Returns
    -------
    PreparedData

    Raises
    ------
    ValueError
        If Y is not 2-D, if ``categories_start`` is not 0 or 1, if
        non-missing response values fall outside the valid category range,
        or if any covariate column is entirely missing.
    """
    # ----------------------------------------------------------------
    # 0. Input validation
    # ----------------------------------------------------------------
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(
            f"Y must be a 2-D array; got shape {Y.shape}."
        )
    if categories_start not in (0, 1):
        raise ValueError(
            f"categories_start must be 0 or 1; got {categories_start}. "
            f"Use categories_start=1 if response categories are 1-based "
            f"(most common) or categories_start=0 if they are 0-based."
        )

    P, I = Y.shape
    R = spec.n_nodes
    K = spec.n_categories

    # Normalize/validate optional entity id labels against Y's dimensions.
    if person_ids is not None:
        person_ids = tuple(person_ids)
        if len(person_ids) != P:
            raise ValueError(
                f"person_ids has length {len(person_ids)} but Y has {P} "
                f"persons (rows). They must match."
            )
    if item_ids is not None:
        item_ids = tuple(item_ids)
        if len(item_ids) != I:
            raise ValueError(
                f"item_ids has length {len(item_ids)} but Y has {I} "
                f"items (columns). They must match."
            )

    # ----------------------------------------------------------------
    # 1. Build observed mask and shift categories
    # ----------------------------------------------------------------
    observed = _build_observed(Y, missing_codes)   # [P, I] bool

    # Safe shift: convert to float first so we can detect NaN
    Y_f = Y.astype(float)
    Y_shifted = Y_f - categories_start   # 0-based internally

    # ----------------------------------------------------------------
    # 2. Validate range on observed cells only
    # ----------------------------------------------------------------
    obs_vals = Y_shifted[observed]
    if len(obs_vals) > 0:
        invalid = (obs_vals < 0) | (obs_vals > K - 1) | np.isnan(obs_vals)
        if np.any(invalid):
            bad_vals = np.unique(obs_vals[invalid]).tolist()
            raise ValueError(
                f"After shifting by categories_start={categories_start}, "
                f"observed responses must be in [0, K-1] = [0, {K-1}]. "
                f"Found out-of-range values: {bad_vals}. "
                f"If your categories start at 1, use categories_start=1."
            )

    # ----------------------------------------------------------------
    # 3. Safe-fill missing cells to 0 for indexing
    # ----------------------------------------------------------------
    Y_safe = Y_shifted.copy()
    Y_safe[~observed] = 0.0
    Y_int = Y_safe.astype(int)   # [P, I], 0-based, no NaN

    # ----------------------------------------------------------------
    # 4. Build Y_star and T for each node
    # ----------------------------------------------------------------
    mapping = spec.mapping   # [K, R] float64; NaN where node not on path

    Y_star  = []
    T       = []
    row_idx = []
    col_idx = []

    for r in range(R):
        # Y_star_r[p, i] = mapping[Y_int[p,i], r] for observed cells
        y_star_r = np.full((P, I), np.nan, dtype=np.float64)
        y_star_r[observed] = mapping[Y_int[observed], r]

        # Structural relevance: cell (p,i) is relevant at node r when
        # its mapping value is not NaN
        structural = ~np.isnan(y_star_r)

        # Combined gate: relevant AND observed
        T_r = structural & observed

        ri, ci = np.where(T_r)

        Y_star.append(y_star_r)
        T.append(T_r)
        row_idx.append(ri)
        col_idx.append(ci)

    # ----------------------------------------------------------------
    # 5. Missing-data report for responses
    # ----------------------------------------------------------------
    n_missing = int(np.sum(~observed))
    if n_missing > 0:
        _print_missing_message(n_missing, P, I, observed, label="responses")

    # ----------------------------------------------------------------
    # 6. Process covariate matrices
    # ----------------------------------------------------------------

    # --- Person covariates X [P, K_x] ---
    if X is not None:
        X_out, X_obs, X_feat_obs, n_miss_X = _process_covariate(
            X,
            expected_leading=(P,),
            axis_names=("n_persons",),
            missing_codes=X_missing_codes,
            label="X (person covariates)",
        )
        if n_miss_X > 0:
            _print_covariate_missing_message(n_miss_X, P, "X (person covariates)")
        X_names_out = _validate_covariate_names(X_names, X_out.shape[1], "X")
    else:
        X_out, X_obs, X_feat_obs, n_miss_X = None, None, None, 0
        if X_names is not None:
            raise ValueError("X_names was given but X is None.")
        X_names_out = None

    # --- Item covariates Q [I, K_q] ---
    if Q is not None:
        Q_out, Q_obs, Q_feat_obs, n_miss_Q = _process_covariate(
            Q,
            expected_leading=(I,),
            axis_names=("n_items",),
            missing_codes=Q_missing_codes,
            label="Q (item covariates)",
        )
        if n_miss_Q > 0:
            _print_covariate_missing_message(n_miss_Q, I, "Q (item covariates)")
        Q_names_out = _validate_covariate_names(Q_names, Q_out.shape[1], "Q")
    else:
        Q_out, Q_obs, Q_feat_obs, n_miss_Q = None, None, None, 0
        if Q_names is not None:
            raise ValueError("Q_names was given but Q is None.")
        Q_names_out = None

    # --- Person×item covariates W [P, I, K_w] ---
    if W is not None:
        W_out, W_obs, W_feat_obs, n_miss_W = _process_covariate(
            W,
            expected_leading=(P, I),
            axis_names=("n_persons", "n_items"),
            missing_codes=W_missing_codes,
            label="W (person×item covariates)",
        )
        if n_miss_W > 0:
            _print_covariate_missing_message(
                n_miss_W, P * I, "W (person×item covariates)"
            )
        W_names_out = _validate_covariate_names(W_names, W_out.shape[2], "W")
    else:
        W_out, W_obs, W_feat_obs, n_miss_W = None, None, None, 0
        if W_names is not None:
            raise ValueError("W_names was given but W is None.")
        W_names_out = None

    # ----------------------------------------------------------------
    # 7. Assemble PreparedData
    # ----------------------------------------------------------------
    return PreparedData(
        Y_star=Y_star,
        T=T,
        row_idx=row_idx,
        col_idx=col_idx,
        observed=observed,
        n_missing=n_missing,
        n_persons=P,
        n_items=I,
        n_nodes=R,
        X=X_out,
        X_observed=X_obs,
        X_feature_observed=X_feat_obs,
        n_missing_X=n_miss_X,
        Q=Q_out,
        Q_observed=Q_obs,
        Q_feature_observed=Q_feat_obs,
        n_missing_Q=n_miss_Q,
        W=W_out,
        W_observed=W_obs,
        W_feature_observed=W_feat_obs,
        n_missing_W=n_miss_W,
        X_names=X_names_out,
        Q_names=Q_names_out,
        W_names=W_names_out,
        person_ids=person_ids,
        item_ids=item_ids,
    )


# ===========================================================================
# Centering covariates
# ===========================================================================

class CenteredCovariates(NamedTuple):
    """Result of ``center_covariates``.

    values   : the centered (and optionally scaled) array, same shape as the
               input, float; missing entries are NaN.
    means    : per-column mean subtracted (0.0 for columns left unchanged).
    scales   : per-column divisor (1.0 unless ``scale=True``).
    centered : per-column flag, True where the column was transformed.
    """
    values:   np.ndarray
    means:    np.ndarray
    scales:   np.ndarray
    centered: np.ndarray


def center_covariates(
    M,
    columns: Sequence | None = None,
    names: Sequence[str] | None = None,
    missing_codes: tuple = (99,),
    scale: bool = False,
    verbose: bool = True,
) -> CenteredCovariates:
    """Center covariates before ``expand_data`` (``load_data(center=...)`` uses it).

    The trait mean and the item-location mean are fixed (at 0 and at the prior
    mean), so an uncentered covariate shifts every trait or location of the
    node it enters and its effect trades off with them.  Centering removes
    that: the coefficients keep their meaning (an effect per unit), and only
    the intercept, which irtree does not estimate separately, changes.

    Parameters
    ----------
    M : array [P, K] (X), [I, K] (Q) or [P, I, K] (W)
        The covariate matrix, laid out as ``expand_data`` expects.  A 1-D
        array is treated as one column.
    columns : sequence of int or str, optional
        Columns to center (indices, or names when ``names`` is given).
        Default: every column that is not constant.
    names : sequence of str, optional
        Column names, for selecting by name and for the printed report.
    missing_codes : tuple
        Sentinel values treated as missing (NaN is always missing).  Missing
        entries are excluded from the mean and returned as NaN, so that
        ``expand_data`` draws them as parameters, centered at 0.
    scale : bool
        Also divide by the standard deviation of the observed values.  The
        default distribution of a missing covariate value is N(0, 1), which
        assumes standardized covariates; scaling changes a coefficient to an
        effect per standard deviation.
    verbose : bool
        Print which columns were centered and which were left unchanged.

    Notes
    -----
    Constant columns (for example a column of ones whose coefficient is a
    node intercept) are never centered: centering would set them to 0.  A
    W covariate is centered over all observed person-item cells.
    """
    A = np.array(M, dtype=float, copy=True)
    one_d = A.ndim == 1
    if one_d:
        A = A[:, None]
    if A.ndim not in (2, 3):
        raise ValueError(
            f"center_covariates expects a [P, K], [I, K] or [P, I, K] array; got shape {A.shape}.")
    K = A.shape[-1]
    if names is not None and len(names) != K:
        raise ValueError(f"names has {len(names)} entries but the array has {K} columns.")
    for code in missing_codes:
        A[A == code] = np.nan

    flat = A.reshape(-1, K)
    obs = ~np.isnan(flat)
    n_obs = obs.sum(axis=0)
    with np.errstate(invalid="ignore"):
        mu = np.where(n_obs > 0, np.nansum(flat, axis=0) / np.maximum(n_obs, 1), np.nan)
        sd = np.sqrt(np.where(n_obs > 1, np.nansum((flat - mu) ** 2, axis=0) / np.maximum(n_obs - 1, 1), 0.0))
    constant = (n_obs == 0) | (sd <= 1e-12 * np.maximum(1.0, np.abs(np.nan_to_num(mu))))

    if columns is None:
        chosen = ~constant
    else:
        chosen = np.zeros(K, dtype=bool)
        for c in columns:
            if isinstance(c, str):
                if names is None or c not in names:
                    raise ValueError(f"unknown covariate column {c!r}; names are {list(names or [])}.")
                chosen[list(names).index(c)] = True
            else:
                chosen[int(c)] = True
        skipped = chosen & constant
        if skipped.any():
            label = [names[j] if names is not None else str(j) for j in np.flatnonzero(skipped)]
            print(f"  center_covariates: constant column(s) {label} left unchanged.")
        chosen &= ~constant

    means = np.where(chosen, mu, 0.0)
    scales = np.where(chosen & scale, sd, 1.0)
    flat[:, chosen] = (flat[:, chosen] - means[chosen]) / scales[chosen]
    out = flat.reshape(A.shape)
    if one_d:
        out = out[:, 0]

    if verbose:
        lab = [names[j] if names is not None else str(j) for j in range(K)]
        done = [lab[j] for j in range(K) if chosen[j]]
        kept = [lab[j] for j in range(K) if not chosen[j]]
        verb = "Centered and scaled" if scale else "Centered"
        print(f"  {verb}: {done if done else 'none'}" + (f"; unchanged: {kept}" if kept else ""))
    return CenteredCovariates(out, means, scales, chosen)


def summarize_prepared(data: PreparedData, spec) -> str:
    """Return a human-readable summary of a PreparedData object.

    Includes per-node statistics and covariate dimensions when present.
    """
    lines = []
    lines.append(
        f"PreparedData: P={data.n_persons}, I={data.n_items}, "
        f"R={data.n_nodes}, missing={data.n_missing}"
    )
    for r in range(data.n_nodes):
        T_r    = data.T[r]
        n_rel  = int(np.sum(T_r))
        y_r    = data.Y_star[r]
        valid  = y_r[T_r]
        mean_y = float(np.mean(valid)) if len(valid) > 0 else float("nan")
        lines.append(
            f"  Node {r + 1}: n_relevant={n_rel}, mean_Y*={mean_y:.3f}"
        )
    if data.X is not None:
        K_x = data.X.shape[1]
        lines.append(
            f"  X (person covariates): K_x={K_x}, "
            f"n_missing={data.n_missing_X}"
        )
    if data.Q is not None:
        K_q = data.Q.shape[1]
        lines.append(
            f"  Q (item covariates): K_q={K_q}, "
            f"n_missing={data.n_missing_Q}"
        )
    if data.W is not None:
        K_w = data.W.shape[2]
        lines.append(
            f"  W (person×item covariates): K_w={K_w}, "
            f"n_missing={data.n_missing_W}"
        )
    return "\n".join(lines)


# ===========================================================================
# Per-node covariate selection
# ===========================================================================
#
# FIRST PRINCIPLES
# ----------------
# Nothing in the Albert-Chib augmented-likelihood derivation requires two
# different nodes to share a design matrix. Node r's full conditional for
# beta_r depends only on (Z_r, T_r, X_r), so giving every covariate node the
# *entire* X matrix would be a software default, not a statistical
# requirement. This module lets each NodeModel
# instance declare its own column subset of X / Q / W via three attributes
# (set on the node model, not stored here -- data_prep.py stays data-only):
#
#     person_covariates : None | True | Sequence[str | int]
#     item_covariates    : None | True | Sequence[str | int]
#     pi_covariates      : None | True | Sequence[str | int]
#
# None -> node uses no covariates of that type.
# True -> node uses every column.
# A list of names (requires *_names passed to expand_data) or integer
# column indices -> exactly those columns, in the given order.
#
# POOLING -- WHAT A "GROUP" IS
# ----------------------------
# A group is defined by exactly ONE criterion: a maximal set of nodes that
# the tree forces to share the *parameter the coefficient explains*. It is
# NOT defined by the covariate, the node model, or tree adjacency.
#
#   * beta (person covariates) -> group == one latent dimension.
#     A group is one entry of spec.dim_to_nodes (the nodes mapped to a
#     single latent variable by node_to_latent). beta is stored per
#     dimension, exactly like theta, because a coefficient's identity is
#     inseparable from the parameter it multiplies: node r's predictor is
#     theta_{dim(r)} - b_i + X_p . beta. If two nodes share theta (same
#     dimension), "the effect of X on this trait" is a single quantity by
#     construction -- giving them separate betas would assert two different
#     relationships to one number. So they must share, and the tree (not
#     the covariate) decides it.
#
#   * kappa (item covariates) -> group == one b-sharing group.
#     A group is one entry of constraints.b_groups_for(R). Same logic, one
#     level down: kappa explains item difficulty, so it pools wherever b
#     itself is pooled.
#
#   * gamma (person x item covariates) -> per node by DEFAULT, but poolable
#     by explicit opt-in via Constraints(gamma_shared=[[...]]). gamma pooling
#     is never forced: gamma multiplies the independent W tensor and stays
#     identified even when theta and b are both shared (verified: two nodes
#     sharing theta AND b still recover distinct gammas from their separate
#     Z regressions). So pooling gamma is a modeling assertion the user makes
#     deliberately, independent of node_to_latent and b_shared. When a gamma
#     group IS declared, its members must all declare pi_covariates and
#     select the identical W columns, for the same coherence reason as
#     beta/kappa: one shared vector has one meaning per position.
#
# RELATIONSHIP TO dim_to_nodes (the key subtlety)
# -----------------------------------------------
# x_groups is NOT a copy of spec.dim_to_nodes. It is dim_to_nodes FILTERED
# to the dimensions that actually have a person covariate declared:
#
#     x_groups = [ dim_to_nodes[d]
#                  for d in range(D)
#                  if at least one node on dimension d declares
#                     person_covariates ]
#
# preserving dim_to_nodes order, with covariate-free dimensions dropped.
# Therefore:
#
#     len(x_groups) = number of dimensions carrying >= 1 covariate
#                   <= len(dim_to_nodes)          (equality iff every
#                                                  dimension has a covariate)
#
# The covariate never creates or reorders groups -- resolve_covariate_indices
# only ever READS the dim_to_nodes / b_groups partition. (q_groups relates to
# constraints.b_groups_for(R) in the identical filtered way.)
#
# THREE COUNTS THAT ARE OFTEN CONFUSED
# ------------------------------------
#     len(x_idx[g])          number of covariate COLUMNS coefficient g uses
#     len(x_groups)          number of distinct COEFFICIENTS (beta vectors)
#     len(x_node_to_group)   number of NODES (== R)
# None of these is derivable from the covariate count alone. In particular,
# two nodes using the *same* covariate column get separate coefficients when
# they are on different dimensions (two x_groups entries whose x_idx happen
# to be equal), and one pooled coefficient when they are on the same
# dimension (a single x_groups entry). Column identity never drives grouping;
# only the dimension structure does.
#
# WORKED EXAMPLES  (X columns named ["rt", "omit", "ses"])
# --------------------------------------------------------
# 1. Independent nodes (no node_to_latent sharing; dim_to_nodes = [[0],[1]]):
#
#        node0 = ProbitRaschNode(person_covariates=["rt", "omit"])
#        node1 = ProbitRaschNode(person_covariates=["ses"])
#
#        x_groups        = [[0], [1]]        # two coefficients
#        x_idx           = [array([0, 1]),   # coef 0 -> rt, omit
#                           array([2])]      # coef 1 -> ses
#        x_node_to_group = [0, 1]            # node 0 -> coef 0; node 1 -> coef 1
#
# 2. Pooled (ers_tree: node_to_latent=(1,2,2), dim_to_nodes = [[0],[1,2]]),
#    covariate only on the shared dimension:
#
#        node0 = ProbitRaschNode()                                # dim 0, none
#        node1 = ProbitRaschNode(person_covariates=["ses"])       # dim 1
#        node2 = ProbitRaschNode(person_covariates=["ses"])       # dim 1
#
#        x_groups        = [[1, 2]]          # ONE coefficient, nodes 1 & 2
#        x_idx           = [array([2])]      # -> ses
#        x_node_to_group = [None, 0, 0]      # dim 0 dropped; nodes 1 & 2 -> coef 0
#
#    Here len(x_groups)=1 while len(dim_to_nodes)=2: the covariate-free
#    dimension 0 contributes no coefficient, so it is filtered out.
#
# 3. Same column, two nodes, SEPARATE coefficients (different dimensions;
#    dim_to_nodes = [[0],[1]]):
#
#        node0 = ProbitRaschNode(person_covariates=["ses"])
#        node1 = ProbitRaschNode(person_covariates=["ses"])       # different dim
#
#        x_groups        = [[0], [1]]        # TWO coefficients ...
#        x_idx           = [array([2]),      # ... both regress on the SAME
#                           array([2])]      #     column (ses)
#        x_node_to_group = [0, 1]            # separate: different g per node
#
#    Contrast with example 2: identical covariate declaration, but the tree's
#    dimension structure -- not the covariate -- decides pool vs. separate.
#
# resolve_covariate_indices() is the single place this gets validated and
# turned into plain index arrays. It does not mutate the NodeModel
# instances (they stay data-free, per the existing design convention) and
# it is meant to be called once, before sampling starts, so a typo in a
# covariate name fails immediately rather than after a chain has been
# running for several minutes.
# ===========================================================================


@dataclass
class CovariateIndex:
    """Resolved, validated covariate column selections for one model.

    Produced once by ``resolve_covariate_indices``. Encodes three separate
    facts about person covariates (and, via the ``q_*`` fields, the same
    three for item covariates):

      1. Which coefficients exist and which nodes each serves -> ``x_groups``
      2. Which X columns each coefficient regresses on          -> ``x_idx``
      3. Given a node, which coefficient builds its offset       -> ``x_node_to_group``

    ``x_groups`` and ``x_idx`` are PARALLEL lists indexed by group ``g``.
    ``x_node_to_group`` is indexed by NODE ``r`` and points back into them.

    What a "group" is
    -----------------
    A group is a maximal set of nodes the tree forces to share the parameter
    the coefficient explains -- NOT anything about the covariate itself:

      * beta group  == one latent dimension  (an entry of spec.dim_to_nodes)
      * kappa group == one b-sharing group   (an entry of
                                               constraints.b_groups_for(R))

    beta is stored per dimension because node r's predictor is
    ``theta_{dim(r)} - b_i + X_p . beta``: nodes sharing theta share "the
    effect of X on that trait" by construction, so they must share beta.

    Relationship to dim_to_nodes
    ----------------------------
    ``x_groups`` is ``spec.dim_to_nodes`` FILTERED to the dimensions that
    actually carry a person covariate (order preserved, covariate-free
    dimensions dropped). Hence::

        len(x_groups) = # dimensions with >= 1 covariate  <=  len(dim_to_nodes)

    with equality iff every dimension has a covariate. The covariate never
    creates or reorders groups; resolution only READS the partition.
    (``q_groups`` relates to ``constraints.b_groups_for(R)`` identically.)

    Three counts not to conflate
    ----------------------------
        len(x_idx[g])          # covariate COLUMNS used by coefficient g
        len(x_groups)          # distinct COEFFICIENTS (beta vectors)
        len(x_node_to_group)   # NODES  (== R)

    Two nodes using the *same* column get SEPARATE coefficients when on
    different dimensions (two groups, equal x_idx) and ONE pooled coefficient
    when on the same dimension (one group). Column identity never drives
    grouping; only dimension structure does.

    Example  (ers_tree, node_to_latent=(1,2,2), dim_to_nodes=[[0],[1,2]];
              only nodes 1 & 2 declare person_covariates=["ses"], col 2)::

        x_groups        = [[1, 2]]          # one pooled coefficient
        x_idx           = [array([2])]      # regressing on "ses"
        x_node_to_group = [None, 0, 0]      # dim 0 dropped; nodes 1&2 -> coef 0

    Attributes
    ----------
    x_groups : list[list[int]]
        Person-covariate groups. Entry ``g`` is the 0-based node indices
        sharing beta coefficient ``g`` (length 1 when independent). A
        filtered sub-list of ``spec.dim_to_nodes``.
    x_idx : list[ndarray[int]]
        Parallel to ``x_groups``: the global X column indices coefficient
        ``g`` uses, in declaration order. ``len(x_idx[g])`` is that group's
        K_x, i.e. the width of ``beta[g]``.
    x_node_to_group : list[int or None]
        Length R. Maps node ``r`` to its index into ``x_groups`` / ``x_idx``,
        or None if node ``r`` has no person covariates. Two nodes with the
        same value pool; with different values, separate; None -> no offset.
    q_groups, q_idx, q_node_to_group
        Same shape and meaning as the ``x_*`` fields, for item covariates
        (Q), grouped by b-sharing group instead of latent dimension.
    w_idx : list[ndarray[int] or None]
        Length R, per node: ``w_idx[r]`` is node ``r``'s global W column
        indices, or None.  A per-node view kept for compatibility; the
        sampler, the chain layout and posterior.py use the group fields
        ``w_groups`` / ``w_group_idx`` / ``w_node_to_group`` below, which
        mirror the ``x_*`` / ``q_*`` layout (gamma pooling via
        ``Constraints(gamma_shared=...)``).
    w_groups : list[list[int]]
        gamma-sharing groups: entry ``g`` is the 0-based node indices
        sharing one gamma vector. With no ``gamma_shared`` declared this is
        the all-singletons partition (one group per covariate node) — the
        per-node default. Unlike beta/kappa, gamma pooling is a modeling
        opt-in, never forced (gamma stays identified even when theta and b
        are shared, because it regresses on the independent W tensor); see
        ``Constraints.gamma_shared``.
    w_group_idx : list[ndarray[int]]
        Parallel to ``w_groups``: the global W columns coefficient ``g``
        uses. Identical across the group's member nodes (enforced).
    w_node_to_group : list[int or None]
        Length R. Maps node ``r`` to its ``w_groups`` index, or None if
        node ``r`` has no person×item covariates.
    """
    x_groups:        list[list[int]] = field(default_factory=list)
    x_idx:            list[np.ndarray] = field(default_factory=list)
    x_node_to_group: list[int | None] = field(default_factory=list)
    q_groups:        list[list[int]] = field(default_factory=list)
    q_idx:            list[np.ndarray] = field(default_factory=list)
    q_node_to_group: list[int | None] = field(default_factory=list)
    w_idx:            list[np.ndarray | None] = field(default_factory=list)
    # gamma grouping (person×item). w_idx above stays per-node (every node's
    # own W columns, or None); the fields below are what the sampler uses,
    # with pooling when Constraints(gamma_shared=...) is set. With no
    # gamma_shared, w_groups is all-singletons and w_node_to_group maps each
    # covariate node to its own group — the same information as w_idx.
    w_groups:        list[list[int]] = field(default_factory=list)
    w_group_idx:     list[np.ndarray] = field(default_factory=list)
    w_node_to_group: list[int | None] = field(default_factory=list)

    def beta_width(self, group: int) -> int:
        """Number of coefficients (K_x) for person-covariate group `group`."""
        return len(self.x_idx[group])

    def kappa_width(self, group: int) -> int:
        """Number of coefficients (K_q) for item-covariate group `group`."""
        return len(self.q_idx[group])


def _resolve_selection(
    selection: None | bool | Sequence,
    names: tuple[str, ...] | None,
    n_cols: int,
    label: str,
) -> np.ndarray | None:
    """Turn one node's raw covariate selection into a global index array.

    Parameters
    ----------
    selection : None | True | False | Sequence[str | int]
        The node model's ``person_covariates`` / ``item_covariates`` /
        ``pi_covariates`` attribute.
    names : tuple[str, ...] or None
        The corresponding ``*_names`` from PreparedData, or None if the
        matrix has no column names.
    n_cols : int
        Total number of columns in the underlying matrix (K_x / K_q / K_w).
    label : str
        Used in error messages, e.g. "node 2 person_covariates".

    Returns
    -------
    ndarray[int] of global column indices, in selection order, or None if
    the node declares no covariates of this type.
    """
    if selection is None or selection is False:
        return None
    if selection is True:
        return np.arange(n_cols, dtype=int)
    if isinstance(selection, (str, bytes)):
        raise TypeError(
            f"{label}: expected a list of column names/indices, got a bare "
            f"string {selection!r}. Wrap it in a list: [{selection!r}]."
        )

    idx: list[int] = []
    for item in selection:
        if isinstance(item, (bool,)):
            raise TypeError(
                f"{label}: entry {item!r} is a bool, not a valid column "
                f"name or index."
            )
        if isinstance(item, (int, np.integer)):
            col = int(item)
            if not (0 <= col < n_cols):
                raise ValueError(
                    f"{label}: column index {col} out of range "
                    f"[0, {n_cols})."
                )
            idx.append(col)
        elif isinstance(item, str):
            if names is None:
                raise ValueError(
                    f"{label}: column name {item!r} given, but no column "
                    f"names were supplied to expand_data() for this "
                    f"matrix. Pass X_names=/Q_names=/W_names= (matching "
                    f"the one this node's covariates come from), or select "
                    f"by integer index instead."
                )
            if item not in names:
                raise ValueError(
                    f"{label}: column name {item!r} not found. "
                    f"Available columns: {list(names)}."
                )
            idx.append(names.index(item))
        else:
            raise TypeError(
                f"{label}: selection entries must be str or int; got "
                f"{type(item).__name__} ({item!r})."
            )

    if len(idx) != len(set(idx)):
        raise ValueError(
            f"{label}: duplicate column selected in {selection!r}."
        )
    return np.array(idx, dtype=int)


def _group_and_validate(
    raw:       list[np.ndarray | None],
    groups_of: list[list[int]],
    kind:      str,
) -> tuple:
    """Fold per-node raw selections into pooled groups, validating consistency.

    Parameters
    ----------
    raw : list[ndarray or None], length R
        Per-node resolved selections from ``_resolve_selection``.
    groups_of : list[list[int]]
        The sharing structure to pool against -- ``spec.dim_to_nodes`` for
        person covariates, or a b-sharing grouping for item covariates.
        Singleton groups (``[[0], [1], [2], ...]``) mean "no pooling",
        and this function degrades to a plain per-node identity mapping.
    kind : str
        "person covariates (latent dimension)" or "item covariates
        (b-sharing group)", used in error messages.

    Returns
    -------
    (groups, idx, node_to_group) -- see CovariateIndex field docs.
    """
    R = len(raw)
    groups: list[list[int]] = []
    idx:    list[np.ndarray] = []
    node_to_group: list[int | None] = [None] * R

    for members in groups_of:
        sels = [(r, raw[r]) for r in members]
        declared = [(r, s) for r, s in sels if s is not None]
        if not declared:
            continue
        if len(declared) != len(members):
            missing = [r for r, s in sels if s is None]
            present = [r for r, s in sels if s is not None]
            raise ValueError(
                f"Nodes {[r + 1 for r in members]} share a {kind}, but "
                f"node(s) {[r + 1 for r in missing]} declare no "
                f"covariates of this type while node(s) "
                f"{[r + 1 for r in present]} do. A covariate that "
                f"explains a shared parameter must be declared "
                f"identically on every node sharing it -- either give "
                f"all of them the same selection, or none."
            )
        first_r, first_sel = declared[0]
        for r, sel in declared[1:]:
            if sel.shape != first_sel.shape or not np.array_equal(sel, first_sel):
                raise ValueError(
                    f"Nodes {[m + 1 for m in members]} share a {kind}, "
                    f"but declare different covariate selections "
                    f"(node {first_r + 1}: {first_sel.tolist()} vs "
                    f"node {r + 1}: {sel.tolist()}). A covariate on a "
                    f"shared parameter must be identical across every "
                    f"node that shares it."
                )
        g = len(groups)
        groups.append(list(members))
        idx.append(first_sel)
        for r in members:
            node_to_group[r] = g

    return groups, idx, node_to_group


def resolve_covariate_indices(
    node_models: list,
    data:        PreparedData,
    spec=None,
    constraints=None,
) -> CovariateIndex:
    """Resolve and validate every node's covariate selection, once.

    Call this before sampling starts (ideally before any parallel chains
    are spawned) so a bad covariate name or an inconsistent pooled
    declaration fails immediately with a clear message, rather than deep
    inside a running chain.

    Parameters
    ----------
    node_models : list of NodeModel
        Reads (via getattr, defaulting to None) each instance's
        ``person_covariates``, ``item_covariates``, ``pi_covariates``.
        Does not mutate the instances.
    data : PreparedData
        Supplies X/Q/W shapes and X_names/Q_names/W_names.
    spec : IRTreeSpec or None
        Supplies ``dim_to_nodes`` for beta pooling. When None, every node
        is treated as its own dimension (no pooling).
    constraints : Constraints or None
        Supplies b-sharing groups (``b_groups_for(R)``) for kappa pooling.
        When None, every node is treated as its own b-group (no pooling).

    Returns
    -------
    CovariateIndex

    Raises
    ------
    ValueError
        Unknown column name, out-of-range index, duplicate column in one
        node's selection, a node declaring covariates when the underlying
        matrix (X/Q/W) is None, or an inconsistent declaration across
        nodes that share a latent dimension / b-group.
    TypeError
        A selection entry that is neither a column name (str) nor an
        integer index.

    Notes
    -----
    The returned ``CovariateIndex`` groups beta per latent dimension
    (``spec.dim_to_nodes``) and kappa per b-sharing group; see that class's
    docstring for the full account of what a "group" is and how
    ``x_groups`` relates to ``dim_to_nodes`` (it is the dimension partition
    FILTERED to dimensions that carry a covariate). The tree's sharing
    structure -- never the covariate -- decides pooling.

    Example
    -------
    >>> # ers_tree: node_to_latent=(1,2,2) -> dim_to_nodes=[[0],[1,2]].
    >>> # Nodes 1 & 2 share latent dim 1, so an identical person covariate
    >>> # on both pools into ONE coefficient; node 0 declares none.
    >>> nms = [ProbitRaschNode(),
    ...        ProbitRaschNode(person_covariates=["ses"]),
    ...        ProbitRaschNode(person_covariates=["ses"])]
    >>> data = expand_data(Y, ers_spec, X=X, X_names=["ses", "age"])
    >>> ci = resolve_covariate_indices(nms, data, spec=ers_spec)
    >>> ci.x_groups          # dim 0 dropped (no covariate); dim 1 kept
    [[1, 2]]
    >>> ci.x_idx             # coefficient regresses on column 0 ("ses")
    [array([0])]
    >>> ci.x_node_to_group   # nodes 1 & 2 -> same group 0 (pooled); node 0 -> None
    [None, 0, 0]
    >>> # Declaring different columns on nodes 1 & 2 would raise (shared dim).
    """
    R = data.n_nodes

    x_raw: list[np.ndarray | None] = [None] * R
    q_raw: list[np.ndarray | None] = [None] * R
    w_raw: list[np.ndarray | None] = [None] * R

    for r, nm in enumerate(node_models):
        person_sel = getattr(nm, "person_covariates", None)
        item_sel   = getattr(nm, "item_covariates", None)
        pi_sel     = getattr(nm, "pi_covariates", None)

        if person_sel is not None and person_sel is not False:
            if data.X is None:
                raise ValueError(
                    f"Node {r + 1} declares person_covariates={person_sel!r} "
                    f"but no X matrix was passed to expand_data()."
                )
            x_raw[r] = _resolve_selection(
                person_sel, data.X_names, data.X.shape[1],
                f"node {r + 1} person_covariates",
            )
        if item_sel is not None and item_sel is not False:
            if data.Q is None:
                raise ValueError(
                    f"Node {r + 1} declares item_covariates={item_sel!r} "
                    f"but no Q matrix was passed to expand_data()."
                )
            q_raw[r] = _resolve_selection(
                item_sel, data.Q_names, data.Q.shape[1],
                f"node {r + 1} item_covariates",
            )
        if pi_sel is not None and pi_sel is not False:
            if data.W is None:
                raise ValueError(
                    f"Node {r + 1} declares pi_covariates={pi_sel!r} "
                    f"but no W tensor was passed to expand_data()."
                )
            w_raw[r] = _resolve_selection(
                pi_sel, data.W_names, data.W.shape[2],
                f"node {r + 1} pi_covariates",
            )

    dim_to_nodes = list(spec.dim_to_nodes) if spec is not None else [[r] for r in range(R)]
    b_groups     = list(constraints.b_groups_for(R)) if constraints is not None else [[r] for r in range(R)]
    gamma_groups_struct = (
        list(constraints.gamma_groups_for(R))
        if constraints is not None else [[r] for r in range(R)]
    )

    x_groups, x_idx, x_node_to_group = _group_and_validate(
        x_raw, dim_to_nodes, "latent dimension (shared theta)",
    )
    q_groups, q_idx, q_node_to_group = _group_and_validate(
        q_raw, b_groups, "b-sharing group (shared item difficulty)",
    )
    # gamma: default per-node (all-singletons partition), pooled only where
    # the user explicitly declared Constraints(gamma_shared=...). Because
    # gamma_groups_struct is a proper partition (singletons for undeclared
    # nodes), _group_and_validate handles both cases uniformly: singleton
    # groups pass through per-node; multi-node groups require every member
    # to declare pi_covariates (partial-declaration error) and to select the
    # IDENTICAL W columns (mismatch error) — the same coherence rule as
    # beta/kappa, for the same reason (one shared vector, one meaning per
    # position).
    w_groups, w_group_idx, w_node_to_group = _group_and_validate(
        w_raw, gamma_groups_struct, "gamma-sharing group (shared person×item effect)",
    )

    return CovariateIndex(
        x_groups=x_groups, x_idx=x_idx, x_node_to_group=x_node_to_group,
        q_groups=q_groups, q_idx=q_idx, q_node_to_group=q_node_to_group,
        w_idx=w_raw,
        w_groups=w_groups, w_group_idx=w_group_idx,
        w_node_to_group=w_node_to_group,
    )
