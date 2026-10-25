# Replication materials: Sections 6 and 7 of the JSS paper

Each paper section has one notebook (or R script) that runs step by step and
times its expensive steps. Every notebook ends by saving its runtime table to
`results/runtime_<notebook>.csv`. All outputs go to a `results/` folder next to
the notebook.

## Order

| Step | File | Language | Paper | Needs |
|---|---|---|---|---|
| 1 | `verbagg/61_verbagg_workflow.ipynb` | Python | 6.1 Verbal aggression workflow | — |
| 2 | `verbagg/72a_lme4_reference.R` | R (lme4) | 7.2 Agreement, part 1 | — |
| 3 | `verbagg/72b_mirt_reference.R` | R (mirt) | 7.2 Agreement, part 2 | — |
| 4 | `verbagg/72_agreement.ipynb` | Python | 7.2 Agreement | outputs of steps 2 and 3 |
| 5 | `sbc/71_sbc.ipynb` | Python | 7.1 Simulation-based calibration | `sbc/sbc_models.py` |

Open each notebook from its own folder (Jupyter's working directory is then the
notebook's folder). The first code cell installs the package from this
repository in editable mode. Run the R scripts from `replication/verbagg`
(`setwd(...)`, then run section by section in RStudio).

## Data

`verbagg/verbagg_long.csv` is the `VerbAgg` data set shipped with the R package
lme4 (316 persons × 24 items, responses no / perhaps / yes, with person and item
covariates), exported unchanged in long format. It is public.

## Reference runtimes

Measured in a 2-core cloud workspace with reduced settings (2 chains × 2,000
iterations; SBC with 2,000 iterations per fit). The notebooks' default
settings are larger, so expect roughly 2 to 4 times these values per fit; the
parallel steps scale with the number of cores.

| Notebook / script | Step | Time (reduced settings) |
|---|---|---|
| 61 | Rasch tree fit / 2PL tree fit | 11 s / 13 s |
| 61 | LOO for both models / HTML report | 4 s / 8 s |
| 72a | lme4 Model A / Model B (one core) | 3.1 min / 0.6 min |
| 72b | mirt M1 / M2 | not measured (run locally) |
| 72 | irtree A / B / B under two other priors / M1 / M2 | 9 / 18 / 36 / 14 / 20 s |
| 71 | one replication: S1 / S2 / S3 | 4.6 / 6.9 / 5.2 s |

SBC at the notebook defaults (N_ITER = 8,000, 300 replications × 3 models) is
the long run: about 20 to 30 s per replication, so roughly 6 hours on one core
or about 1 hour on 7 workers. Step 3 of `71_sbc.ipynb` times one replication
per model on your machine and prints the projection before the full run; the
full run writes after every replication and resumes if interrupted.
