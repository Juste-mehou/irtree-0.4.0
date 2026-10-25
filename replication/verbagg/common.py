"""Shared data preparation for the verbal aggression scripts (Sections 6.1, 7.2).

The data are lme4's ``VerbAgg`` (316 persons, 24 items, responses
no / perhaps / yes), exported by ``fit_lme4.R``'s companion export to
``verbagg_long.csv``.  The covariate coding matches ``fit_lme4.R`` exactly:

* person covariates X: ``anger`` (Anger, centered) and ``male`` (Gender == M,
  centered);
* item covariates Q: ``scold``, ``shout`` (behavior type; curse is the
  reference), ``self`` (situation: self to blame), ``do`` (mode: do vs want),
  each a centered indicator.  ``intercept`` is a column of ones: because the
  item locations have prior mean 0, its coefficient is the node intercept
  (lme4's ``node`` fixed effect).

The tree is De Boeck and Partchev's (2012) linear tree: node 1 separates
"no" (0) from "perhaps"/"yes" (1); node 2 separates "perhaps" (0) from
"yes" (1) and is reached only by "perhaps"/"yes".
"""
from pathlib import Path

import numpy as np
import pandas as pd

import irtree

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CATEGORIES = ["no", "perhaps", "yes"]


def linear_tree():
    """Mapping matrix of the linear tree, categories coded 0, 1, 2."""
    return irtree.IRTreeSpec(mapping=np.array([[0, np.nan],
                                               [1, 0],
                                               [1, 1]]))


def load():
    """Return Y (P x I, coded 0-2), X, X_names, Q, Q_names, items, persons."""
    va = pd.read_csv(HERE / "verbagg_long.csv")
    items = list(dict.fromkeys(va["item"]))                 # lme4 level order
    persons = sorted(va["id"].unique())
    va["y"] = va["resp"].map({c: k for k, c in enumerate(CATEGORIES)})
    Y = (va.pivot(index="id", columns="item", values="y")
           .loc[persons, items].to_numpy(dtype=float))

    pers = va.drop_duplicates("id").set_index("id").loc[persons]
    X = np.column_stack([pers["Anger"] - pers["Anger"].mean(),
                         (pers["Gender"] == "M") - (pers["Gender"] == "M").mean()])
    X_names = ["anger", "male"]

    it = va.drop_duplicates("item").set_index("item").loc[items]
    ind = np.column_stack([it["btype"] == "scold", it["btype"] == "shout",
                           it["situ"] == "self", it["mode"] == "do"]).astype(float)
    Q = np.column_stack([np.ones(len(items)), ind - ind.mean(axis=0)])
    Q_names = ["intercept", "scold", "shout", "self", "do"]
    return Y, X, X_names, Q, Q_names, items, persons
