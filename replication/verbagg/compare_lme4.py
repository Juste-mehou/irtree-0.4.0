"""Section 7.2: irtree against lme4::glmer on the verbal aggression linear tree.

Fits the two models of ``fit_lme4.R`` with logistic Rasch nodes and compares
posterior means and SDs with glmer's estimates and standard errors.

Model A  item-by-node locations, nearly flat prior N(0, 10) (glmer: fixed
         effects item:node; glmer's coefficient is -b).
Model B  random item locations (default hierarchical prior), item covariates
         with a node intercept, person covariates; coefficients N(0, 10).

Run ``Rscript replication/verbagg/fit_lme4.R`` first, then, from the
repository root:  python replication/verbagg/compare_lme4.py
Output: results/irtree_model{A,B}.csv, results/comparison_lme4.csv
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import irtree                                        # noqa: E402
from common import RESULTS, linear_tree, load        # noqa: E402

N_ITER, N_CHAINS, SEED = 4000, 4, 2026
VAGUE = irtree.NormalPrior(mean=0.0, variance=10.0)


def fit_model(model):
    Y, X, Xn, Q, Qn, items, persons = load()
    spec = linear_tree()
    cov = model == "B"
    data = irtree.expand_data(Y, spec, categories_start=0,
                              X=X if cov else None, X_names=Xn if cov else None,
                              Q=Q if cov else None, Q_names=Qn if cov else None)
    nodes = [irtree.LogitRaschNode(person_covariates=cov or None,
                                   item_covariates=cov or None) for _ in range(2)]
    priors = irtree.build_default_priors(spec, nodes)   # theta correlation: LKJ(1)
    if model == "A":
        priors.set("b", VAGUE)
    else:
        priors.set("beta", VAGUE)
        priors.set("kappa", VAGUE)
    t0 = time.time()
    fit = irtree.irtree_fit(data, spec, nodes, priors=priors, n_iter=N_ITER,
                            n_chains=N_CHAINS, seed=SEED, progress=False)
    print(f"Model {model}: {time.time() - t0:.0f} s")
    df = fit.summary_df()
    df.to_csv(RESULTS / f"irtree_model{model}.csv")
    return df, items


def lme4_key(model, name, items):
    """Map an irtree parameter name to (glmer term, sign)."""
    node = {"1": "n1", "2": "n2"}
    p = name.split("_")
    if model == "A" and p[0] == "b":
        return f"item{items[int(p[2]) - 1]}:node{node[p[1]]}", -1.0
    if p[0] == "sigma2":
        return f"var_id_node{node[p[1]]}", 1.0
    if name == "rho_1_2":
        return "cor_id_noden1_noden2", 1.0
    if model == "B" and p[0] == "b.variance":
        return ("var_item_d1", "var_item.1_d2")[int(p[1]) - 1], 1.0
    if model == "B" and p[0] in ("beta", "kappa"):
        cov = "_".join(p[2:])
        if cov == "intercept":
            return f"node{node[p[1]]}", 1.0
        return f"node{node[p[1]]}:{cov}", 1.0
    return None, None


def main():
    rows = []
    for model in ("A", "B"):
        ref = pd.read_csv(RESULTS / f"lme4_model{model}.csv").set_index("term")
        df, items = fit_model(model)
        for name, row in zip(df.index, df.itertuples()):
            term, sign = lme4_key(model, str(name), items)
            if term is None or term not in ref.index:
                continue
            rows.append(dict(model=model, parameter=name, glmer_term=term,
                             irtree_mean=sign * row.mean, irtree_sd=row.sd,
                             glmer_est=ref.loc[term, "estimate"],
                             glmer_se=ref.loc[term, "se"]))
    out = pd.DataFrame(rows)
    out["diff"] = out["irtree_mean"] - out["glmer_est"]
    out["diff_in_se"] = out["diff"] / out["glmer_se"]
    out.to_csv(RESULTS / "comparison_lme4.csv", index=False)
    pd.set_option("display.width", 160)
    for model, g in out.groupby("model"):
        loc = g[g["glmer_se"].notna()]
        print(f"\nModel {model}: {len(g)} parameters compared; "
              f"location/coefficient max |diff| = {loc['diff'].abs().max():.3f}, "
              f"max |diff|/SE = {loc['diff_in_se'].abs().max():.2f}, "
              f"corr = {np.corrcoef(loc['irtree_mean'], loc['glmer_est'])[0, 1]:.4f}")
        print(g[g["glmer_se"].isna()][["parameter", "irtree_mean", "irtree_sd", "glmer_est"]]
              .to_string(index=False))
        if model == "B":
            print(loc[["parameter", "irtree_mean", "irtree_sd", "glmer_est", "glmer_se"]]
                  .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
