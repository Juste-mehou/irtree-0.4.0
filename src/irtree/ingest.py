"""
irtree.ingest
=============
Layer 1.5 — Data ingestion: raw files / frames → aligned arrays → PreparedData.

This module is the *front door* that sits above ``data_prep.expand_data``.
``expand_data`` assumes clean, positionally-aligned numpy arrays and does no
file loading, no joining, and no identity checking: it matches X to persons,
Q to items, and W to both purely by array position. ``ingest.load_data``
establishes that alignment by joining on explicit ``person_id`` / ``item_id``
keys, so that by the time ``expand_data`` runs, positional alignment is correct
*by construction*.

Design decisions:

  * Canonical order is FIRST-SEEN in the responses table (not sorted), for
    both persons and items. This preserves the researcher's item numbering
    and fixes the integer positions the sampler's parameter names carry.
  * Optional patsy formula expansion per grain (extra: ``irtree[formula]``).
  * pandas is a hard dependency of THIS module only; the core stays
    numpy/scipy-only.
  * ``person_ids`` / ``item_ids`` are carried through to ``PreparedData`` so
    results can be relabelled by the researcher's own ids at display time.

THE THREE FORCING RULES
-----------------------
  1. ``person_id`` and ``item_id`` are mandatory wherever their grain appears.
     Positional alignment is never accepted here (for already-aligned numpy
     arrays, call ``expand_data`` directly).
  2. The response table is the spine: it defines the canonical person index and
     item index; every covariate is reindexed / pivoted onto it.
  3. W (cell covariates) is always supplied LONG — one row per (person, item),
     one column per covariate — never as a wide item×covariate block. Item
     identity is therefore always a real column, never parsed from a header.

TWO INPUT MODES
---------------
  Separate tables (normalized): a response table plus up to three covariate
    tables (person X, item Q, cell W), joined on ids. Use the
    ``person_covariates`` / ``item_covariates`` / ``cell_covariates`` arguments.

  Single long table (denormalized): one table whose rows are (person, item)
    cells, carrying the response plus X columns (constant within person),
    Q columns (constant within item) and W columns (per cell). Declare which
    columns are which via ``roles={"X": [...], "Q": [...], "W": [...]}``.

Both canonicalize to the same arrays and the same ``PreparedData``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import numpy as np

try:
    import pandas as pd
except ImportError as exc:                      # pragma: no cover
    raise ImportError(
        "irtree.ingest requires pandas. Install it with `pip install pandas`, "
        "or build PreparedData yourself via irtree.expand_data (the raw-array "
        "path) if you want to avoid the pandas dependency."
    ) from exc


TableLike = Union[str, "pd.DataFrame"]


# ===========================================================================
# Small reporting helper (matches data_prep.py's print-based convention)
# ===========================================================================

def _report(msg: str, verbose: bool) -> None:
    if verbose:
        print(msg)


def _sample_ids(ids: Sequence, k: int = 5) -> str:
    ids = list(ids)
    shown = ", ".join(repr(x) for x in ids[:k])
    return shown + (", ..." if len(ids) > k else "")


# ===========================================================================
# Transport: coerce any accepted source to a DataFrame
# ===========================================================================

def _as_dataframe(src: TableLike, label: str) -> pd.DataFrame:
    """Coerce a path or DataFrame to a DataFrame. Does not copy a DataFrame."""
    if isinstance(src, pd.DataFrame):
        return src
    if isinstance(src, str):
        # Sniff separator: .tsv → tab, else default comma. pandas' python engine
        # with sep=None would auto-sniff, but we keep it explicit and fast.
        sep = "\t" if src.lower().endswith((".tsv", ".tab")) else ","
        return pd.read_csv(src, sep=sep)
    raise TypeError(
        f"{label} must be a file path (str) or a pandas DataFrame; "
        f"got {type(src).__name__}."
    )


def _require_columns(df: pd.DataFrame, cols: Sequence[str], label: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label}: required column(s) {missing} not found. "
            f"Available columns: {list(df.columns)}."
        )


def _first_seen(values) -> list:
    """Order of first appearance, duplicates removed. pandas.unique preserves it."""
    return list(pd.unique(np.asarray(values)))


# ===========================================================================
# Optional formula expansion (patsy)
# ===========================================================================

def _expand_formula(
    df: pd.DataFrame,
    formula: str,
    label: str,
    drop_intercept: bool,
) -> pd.DataFrame:
    """Expand a patsy formula over ``df``'s columns into a numeric design frame.

    Returns a DataFrame with the SAME index as ``df`` (so it can be reindexed
    onto the canonical order afterwards) and one column per generated term.

    Missing values are PRESERVED as NaN (NA_action disabled) rather than
    dropped, so persons/items/cells with a missing covariate survive as NaN
    rows for the sampler to impute. Caveat: a missing value in a CATEGORICAL
    term cannot be dummy-coded and will raise; encode categorical missingness
    explicitly (an explicit level) or pre-expand that factor.

    The automatically-added intercept is dropped by default (``drop_intercept``)
    because the latent trait mean and the item difficulty already absorb a
    constant; a covariate-design intercept would be redundant with them. Pass
    ``drop_intercept=False`` to keep it.
    """
    try:
        from patsy import NAAction, dmatrix
    except ImportError as exc:                  # pragma: no cover
        raise ImportError(
            f"A formula was given for {label} but patsy is not installed. "
            f"Install it with `pip install irtree[formula]` (or `pip install patsy`), "
            f"or pass an already-numeric covariate table with no formula."
        ) from exc

    design = dmatrix(
        formula, df,
        return_type="dataframe",
        NA_action=NAAction(NA_types=[]),        # keep NaN rows, don't drop
    )
    if drop_intercept and "Intercept" in design.columns:
        design = design.drop(columns=["Intercept"])
    return design


# ===========================================================================
# Response readers → Y [P, I] plus canonical person/item order
# ===========================================================================

def _responses_wide(
    df: pd.DataFrame,
    person_id: str,
    item_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, list, list]:
    """Wide responses: one row per person, headers are item ids.

    By default every column except ``person_id`` is an item. Pass
    ``item_columns`` to name the item columns explicitly when the wide file
    also carries non-item columns (a group variable, a stray index), or to make
    the item-id vocabulary explicit so it matches the Q / W tables.
    """
    _require_columns(df, [person_id], "responses (wide)")

    if df[person_id].duplicated().any():
        dups = df[person_id][df[person_id].duplicated()].tolist()
        raise ValueError(
            f"responses (wide): person_id {_sample_ids(dups)} appear on more "
            f"than one row. In wide layout each person is exactly one row."
        )

    if item_columns is not None:
        item_cols = list(item_columns)
        _require_columns(df, item_cols, "responses (wide) item_columns")
    else:
        item_cols = [c for c in df.columns if c != person_id]
    if not item_cols:
        raise ValueError("responses (wide): no item columns found besides person_id.")
    person_order = _first_seen(df[person_id])
    item_order   = list(item_cols)              # item ids ARE the headers (first-seen = left→right)

    Y_df = df.set_index(person_id).loc[person_order, item_cols]
    Y    = _to_float(Y_df.to_numpy(), "responses")
    return Y, person_order, item_order


def _responses_long(
    df: pd.DataFrame,
    person_id: str,
    item_id: str,
    response_col: str,
) -> tuple[np.ndarray, list, list]:
    """Long responses: (person_id, item_id, response) triples."""
    _require_columns(df, [person_id, item_id, response_col], "responses (long)")

    dup = df.duplicated([person_id, item_id])
    if dup.any():
        bad = df.loc[dup, [person_id, item_id]].apply(tuple, axis=1).tolist()
        raise ValueError(
            f"responses (long): duplicate (person_id, item_id) pair(s) "
            f"{_sample_ids(bad)}. Each (person, item) cell may appear once."
        )

    person_order = _first_seen(df[person_id])
    item_order   = _first_seen(df[item_id])

    piv = df.pivot(index=person_id, columns=item_id, values=response_col)
    piv = piv.reindex(index=person_order, columns=item_order)
    Y   = _to_float(piv.to_numpy(), "responses")
    return Y, person_order, item_order


def _to_float(arr: np.ndarray, label: str) -> np.ndarray:
    try:
        return arr.astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"{label}: values must be numeric category codes (missing may be "
            f"NaN or a sentinel). Could not convert to float. If your "
            f"categories are strings, recode them to integers first."
        ) from exc


# ===========================================================================
# Covariate processors → aligned arrays + names
# ===========================================================================

def _covariate_columns(df: pd.DataFrame, key_cols: Sequence[str]) -> list[str]:
    return [c for c in df.columns if c not in key_cols]


def _norm_key(x):
    """Canonical comparable form of an id, robust to int / float-int / whitespace.

    The common conflict is a wide response file (headers are strings) joined
    against a covariate whose ``item_id`` column pandas read as int64: ``"3"``
    vs ``3``. Those mean the same item, so we compare on a normalized form
    (integers and integer-valued floats collapse to their digits; strings are
    stripped). This resolves pure representation differences WITHOUT changing
    meaning; genuine label mismatches still fail to match and are caught below.
    """
    if x is None:
        return None
    if isinstance(x, (float, np.floating)):
        xf = float(x)
        if np.isfinite(xf) and xf.is_integer():
            return str(int(xf))
        return repr(xf)
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return str(x).strip()


def _reconcile_keys(
    df: pd.DataFrame,
    key: str,
    canonical_ids,
    label: str,
    verbose: bool,
    report_uncovered: bool = True,
) -> pd.DataFrame:
    """Align a covariate table's ``key`` column onto the response-side ids.

    Matching is on the normalized key form, so an int/str or whitespace
    difference between the covariate's ``key`` and the (possibly header-
    inferred) response ids does not cause silent misalignment. Matched rows'
    ``key`` is rewritten to the canonical response-side id so downstream
    reindex/pivot aligns on it.

    Raises when NONE of the covariate's keys match any response id (a genuine
    id-vocabulary conflict, e.g. wide response headers ``i1, i2`` vs a Q file
    keyed ``item01, item02``). Covariate keys matching no response id are
    dropped with a report; response ids with no covariate row are imputed and
    (when ``report_uncovered``) reported — suppressed for the cell grain, where
    sparsity across cells is normal rather than a problem.
    """
    canon = list(canonical_ids)
    canon_norm: dict = {}
    for c in canon:
        canon_norm.setdefault(_norm_key(c), c)

    kn = df[key].map(_norm_key)
    in_canon = kn.isin(set(canon_norm))
    matched = df[in_canon].copy()
    matched[key] = kn[in_canon].map(canon_norm).values

    matched_canon = set(matched[key])
    if len(matched_canon) == 0:
        raise ValueError(
            f"{label}: none of its {key!r} values match any response "
            f"{key!r}. Response side sample: {_sample_ids(canon)}; {label} "
            f"side sample: {_sample_ids(_first_seen(df[key]))}. (Ids were "
            f"compared ignoring int/float/whitespace differences, so this is a "
            f"genuine labelling mismatch — rename so the two sides agree, or "
            f"in wide mode name the item columns with `item_columns=`.)"
        )

    extras = _first_seen(df.loc[~in_canon, key])
    if len(extras):
        _report(
            f"{label}: {len(extras)} {key}(s) not among the response ids were "
            f"dropped [{_sample_ids(extras)}].",
            verbose,
        )
    if report_uncovered:
        uncovered = [c for c in canon if c not in matched_canon]
        if uncovered:
            _report(
                f"{label}: {len(uncovered)} response id(s) have no {label} row; "
                f"their values will be imputed [{_sample_ids(uncovered)}].",
                verbose,
            )
    return matched


def _process_person(
    src: TableLike,
    person_id: str,
    person_order: list,
    formula: str | None,
    drop_intercept: bool,
    verbose: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Person covariate table → X [P, K_x] reindexed to person_order, + names.

    Persons present in the responses but absent from X become all-NaN rows
    (the sampler imputes them). Persons present in X but absent from the
    responses are dropped with a report.
    """
    df = _as_dataframe(src, "person_covariates")
    _require_columns(df, [person_id], "person_covariates")
    df = _reconcile_keys(df, person_id, person_order, "person_covariates", verbose)
    if df[person_id].duplicated().any():
        dups = df[person_id][df[person_id].duplicated()].tolist()
        raise ValueError(
            f"person_covariates: duplicate person_id {_sample_ids(dups)} "
            f"(after id normalization). Person covariates are one row per person."
        )
    df = df.set_index(person_id)

    if formula is not None:
        design = _expand_formula(df, formula, "person_covariates", drop_intercept)
    else:
        design = df[_covariate_columns(df, [])]

    design = design.reindex(person_order)       # NaN rows for absent persons
    names  = tuple(str(c) for c in design.columns)
    X      = design.to_numpy(dtype=float)
    return X, names


def _process_item(
    src: TableLike,
    item_id: str,
    item_order: list,
    formula: str | None,
    drop_intercept: bool,
    verbose: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Item covariate table → Q [I, K_q] reindexed to item_order, + names."""
    df = _as_dataframe(src, "item_covariates")
    _require_columns(df, [item_id], "item_covariates")
    df = _reconcile_keys(df, item_id, item_order, "item_covariates", verbose)
    if df[item_id].duplicated().any():
        dups = df[item_id][df[item_id].duplicated()].tolist()
        raise ValueError(
            f"item_covariates: duplicate item_id {_sample_ids(dups)} "
            f"(after id normalization). Item covariates are one row per item."
        )
    df = df.set_index(item_id)

    if formula is not None:
        design = _expand_formula(df, formula, "item_covariates", drop_intercept)
    else:
        design = df[_covariate_columns(df, [])]

    design = design.reindex(item_order)
    names  = tuple(str(c) for c in design.columns)
    Q      = design.to_numpy(dtype=float)
    return Q, names


def _process_cell(
    src: TableLike,
    person_id: str,
    item_id: str,
    person_order: list,
    item_order: list,
    formula: str | None,
    drop_intercept: bool,
    verbose: bool,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Long cell covariate table → W [P, I, K_w] pivoted on both ids, + names.

    Cells present in the responses but absent from W become all-NaN along the
    covariate axis (sampler imputes). Rows whose person OR item is absent from
    the responses are dropped with a report.
    """
    df = _as_dataframe(src, "cell_covariates")
    _require_columns(df, [person_id, item_id], "cell_covariates")

    # Reconcile each axis onto the response-side ids (normalized match). No
    # per-axis duplicate check: a person/item legitimately repeats across cells.
    df = _reconcile_keys(df, person_id, person_order, "cell_covariates",
                         verbose, report_uncovered=False)
    df = _reconcile_keys(df, item_id,  item_order,   "cell_covariates",
                         verbose, report_uncovered=False)

    dup = df.duplicated([person_id, item_id])
    if dup.any():
        bad = df.loc[dup, [person_id, item_id]].apply(tuple, axis=1).tolist()
        raise ValueError(
            f"cell_covariates: duplicate (person_id, item_id) pair(s) "
            f"{_sample_ids(bad)} (after id normalization). Each cell may appear once."
        )

    keys = df[[person_id, item_id]].reset_index(drop=True)
    body = df.drop(columns=[person_id, item_id]).reset_index(drop=True)

    if formula is not None:
        design = _expand_formula(body, formula, "cell_covariates", drop_intercept)
    else:
        design = body

    names = tuple(str(c) for c in design.columns)
    if not names:
        raise ValueError("cell_covariates: no covariate columns found besides the ids.")

    # Pivot each covariate column to [P, I], then stack to [P, I, K_w].
    keyed = pd.concat([keys, design.reset_index(drop=True)], axis=1)
    planes = []
    for name in design.columns:
        piv = keyed.pivot(index=person_id, columns=item_id, values=name)
        piv = piv.reindex(index=person_order, columns=item_order)
        planes.append(piv.to_numpy(dtype=float))
    W = np.stack(planes, axis=2)                 # [P, I, K_w]
    return W, names


# ===========================================================================
# Single-table (denormalized) split
# ===========================================================================

def _check_constant_within(
    df: pd.DataFrame,
    key: str,
    cols: Sequence[str],
    grain: str,
) -> None:
    """Verify each column in ``cols`` has one value per ``key`` (ignoring NaN)."""
    for c in cols:
        counts = df.groupby(key)[c].nunique(dropna=True)
        bad = counts[counts > 1].index.tolist()
        if bad:
            raise ValueError(
                f"Single-table mode: column {c!r} is declared a {grain} "
                f"covariate but is not constant within {key} (it varies for "
                f"{key} {_sample_ids(bad)}). This is a merge/duplication error: "
                f"a {grain}-grain attribute must repeat identically down every "
                f"row of the same {key}."
            )


def _split_single_table(
    df: pd.DataFrame,
    person_id: str,
    item_id: str,
    response_col: str,
    roles: dict[str, Sequence[str]],
) -> tuple[pd.DataFrame, pd.DataFrame | None,
           pd.DataFrame | None, pd.DataFrame | None]:
    """Split one long (person, item, response, …) table into responses + X/Q/W.

    Returns (responses_long, X_df, Q_df, W_df), each keyed and ready for the
    separate-table processors. Covariate-free grains come back as None.
    """
    _require_columns(df, [person_id, item_id, response_col], "responses (single-table)")

    x_cols = list(roles.get("X", []))
    q_cols = list(roles.get("Q", []))
    w_cols = list(roles.get("W", []))
    all_declared = x_cols + q_cols + w_cols
    _require_columns(df, all_declared, "responses (single-table roles)")

    dup = {c for c in all_declared if all_declared.count(c) > 1}
    if dup:
        raise ValueError(
            f"Single-table roles: column(s) {sorted(dup)} assigned to more "
            f"than one grain. Each covariate column belongs to exactly one of "
            f"X / Q / W."
        )

    responses_long = df[[person_id, item_id, response_col]]

    X_df = None
    if x_cols:
        _check_constant_within(df, person_id, x_cols, "person (X)")
        X_df = df.groupby(person_id, sort=False)[x_cols].first().reset_index()

    Q_df = None
    if q_cols:
        _check_constant_within(df, item_id, q_cols, "item (Q)")
        Q_df = df.groupby(item_id, sort=False)[q_cols].first().reset_index()

    W_df = None
    if w_cols:
        W_df = df[[person_id, item_id] + w_cols]

    return responses_long, X_df, Q_df, W_df


# ===========================================================================
# Public API
# ===========================================================================

def load_data(
    responses: TableLike,
    *,
    spec,
    person_id:    str = "person_id",
    item_id:      str = "item_id",
    response_col: str = "response",
    responses_layout: str = "auto",            # "auto" | "wide" | "long"
    item_columns: Sequence[str] | None = None,  # wide only: name the item cols
    # --- separate-table mode ---
    person_covariates: TableLike | None = None,
    item_covariates:   TableLike | None = None,
    cell_covariates:   TableLike | None = None,
    # --- single-table mode ---
    roles: dict[str, Sequence[str]] | None = None,
    # --- optional formulas per grain (needs irtree[formula]) ---
    formulas: dict[str, str] | None = None,
    drop_intercept: bool = True,
    # --- passthrough to expand_data ---
    categories_start: int = 0,
    missing_codes: tuple = (99,),
    covariate_missing_codes: tuple = (99,),
    # --- centering (see data_prep.center_covariates) ---
    center: bool | dict = False,
    scale: bool = False,
    verbose: bool = True,
):
    """Load a researcher's datasets into a validated, aligned ``PreparedData``.

    Joins covariates onto the response table by ``person_id`` / ``item_id``
    (never by position), orders persons and items by first appearance in the
    responses, pivots cell covariates from long to a [P, I, K_w] tensor,
    optionally expands patsy formulas, and hands the aligned arrays to
    ``expand_data`` together with the id vectors.

    Parameters
    ----------
    responses : path or DataFrame
        Wide (one row per person, headers are item ids) or long
        (person_id, item_id, response). In single-table mode (``roles`` given)
        this is the one long table carrying the covariate columns too.
    spec : IRTreeSpec
        Tree specification, forwarded to ``expand_data``. Required.
    person_id, item_id : str
        Key column names. ``item_id`` is unused for wide responses (the headers
        are the item ids) unless cell covariates or single-table mode need it.
    response_col : str
        Response column name for long / single-table responses.
    responses_layout : {"auto", "wide", "long"}
        "auto" picks long when both ``item_id`` and ``response_col`` are columns
        of ``responses``, otherwise wide. ``roles`` forces single-table (long).
    item_columns : sequence of str, optional
        Wide layout only. The response columns that are items (their headers
        become the item ids). Use it when the wide file also holds non-item
        columns, or to pin the item-id vocabulary so it matches Q / W. Defaults
        to every column except ``person_id``.
    person_covariates, item_covariates, cell_covariates : path or DataFrame
        Separate-table covariates (X, Q, W). Cell covariates must be long.
    roles : dict, optional
        Single-table mode: ``{"X": [...], "Q": [...], "W": [...]}`` naming which
        columns of ``responses`` are person / item / cell covariates.
    formulas : dict, optional
        ``{"X": "...", "Q": "...", "W": "..."}`` patsy formulas per grain.
    drop_intercept : bool
        Drop patsy's automatic intercept from covariate designs (default True;
        the latent mean and item difficulty already carry a constant).
    categories_start, missing_codes : forwarded to ``expand_data``.
    covariate_missing_codes : tuple
        Sentinel(s) treated as missing in X/Q/W (NaN is always missing).
    center : bool or dict
        Center the covariates after alignment (``center_covariates``), so that
        they do not shift the traits or item locations whose means are fixed.
        ``True`` centers every non-constant column of X, Q and W; a dict such
        as ``{"X": True, "Q": ["load"], "W": False}`` chooses per grain (a list
        names the columns).  Constant columns, such as an intercept, are left
        unchanged.  Default False (covariates used as given).  For numpy arrays
        passed straight to ``expand_data``, call ``center_covariates`` first.
    scale : bool
        With ``center``, also divide the centered columns by their standard
        deviation (the default N(0, 1) distribution of a missing covariate
        value assumes standardized covariates).
    verbose : bool
        Print a note when covariate rows with ids absent from the responses
        are dropped, and when response ids have no covariate row (their
        values will be imputed).  Default True.

    Returns
    -------
    PreparedData
        With ``person_ids`` / ``item_ids`` populated in canonical order.
    """
    if spec is None:
        raise ValueError("load_data requires a tree `spec` to expand the data.")
    if roles is not None and any(
        c is not None for c in (person_covariates, item_covariates, cell_covariates)
    ):
        raise ValueError(
            "Pass EITHER single-table `roles` OR separate "
            "`person_/item_/cell_covariates`, not both."
        )

    formulas = formulas or {}
    f_x, f_q, f_w = formulas.get("X"), formulas.get("Q"), formulas.get("W")

    # ---- Responses → Y + canonical order -----------------------------------
    resp_df = _as_dataframe(responses, "responses")

    if roles is not None:
        # Single-table (always long) → split into responses + X/Q/W pieces.
        resp_long, X_df, Q_df, W_df = _split_single_table(
            resp_df, person_id, item_id, response_col, roles
        )
        Y, person_order, item_order = _responses_long(
            resp_long, person_id, item_id, response_col
        )
        person_covariates = X_df
        item_covariates   = Q_df
        cell_covariates   = W_df
    else:
        layout = responses_layout
        if layout == "auto":
            layout = "long" if (item_id in resp_df.columns
                                and response_col in resp_df.columns) else "wide"
        if layout == "wide":
            Y, person_order, item_order = _responses_wide(
                resp_df, person_id, item_columns=item_columns)
        elif layout == "long":
            Y, person_order, item_order = _responses_long(
                resp_df, person_id, item_id, response_col
            )
        else:
            raise ValueError(
                f"responses_layout must be 'auto', 'wide' or 'long'; got {layout!r}."
            )

    _report(
        f"Ingest: P={len(person_order)} persons, I={len(item_order)} items "
        f"(first-seen order).",
        verbose,
    )

    # ---- Covariates → aligned arrays ---------------------------------------
    X = Q = W = None
    X_names = Q_names = W_names = None

    if person_covariates is not None:
        X, X_names = _process_person(
            person_covariates, person_id, person_order, f_x, drop_intercept, verbose
        )
    if item_covariates is not None:
        Q, Q_names = _process_item(
            item_covariates, item_id, item_order, f_q, drop_intercept, verbose
        )
    if cell_covariates is not None:
        W, W_names = _process_cell(
            cell_covariates, person_id, item_id,
            person_order, item_order, f_w, drop_intercept, verbose
        )

    # ---- Optional centering ------------------------------------------------
    from irtree.data_prep import center_covariates, expand_data

    if center:
        if center is True:
            plan = {"X": True, "Q": True, "W": True}
        elif isinstance(center, dict):
            unknown = set(center) - {"X", "Q", "W"}
            if unknown:
                raise ValueError(f"center keys must be 'X', 'Q' or 'W'; got {sorted(unknown)}.")
            plan = center
        else:
            raise TypeError("center must be True, False, or a dict keyed by 'X', 'Q', 'W'.")
        grains = {"X": (X, X_names), "Q": (Q, Q_names), "W": (W, W_names)}
        done = {}
        for g, (M, nm_) in grains.items():
            sel = plan.get(g, False)
            if M is None or sel is False:
                continue
            if verbose:
                print(f"  {g}:", end="")
            res = center_covariates(M, columns=None if sel is True else sel, names=nm_,
                                    missing_codes=covariate_missing_codes, scale=scale,
                                    verbose=verbose)
            done[g] = res.values
        X, Q, W = done.get("X", X), done.get("Q", Q), done.get("W", W)

    # ---- Hand off to expand_data -------------------------------------------

    person_ids = tuple(person_order)
    item_ids   = tuple(item_order)

    return expand_data(
        Y, spec,
        categories_start=categories_start,
        missing_codes=missing_codes,
        X=X, Q=Q, W=W,
        X_missing_codes=covariate_missing_codes,
        Q_missing_codes=covariate_missing_codes,
        W_missing_codes=covariate_missing_codes,
        X_names=X_names, Q_names=Q_names, W_names=W_names,
        person_ids=person_ids, item_ids=item_ids,
    )
