# Replication materials: Sections 6 and 7 of the JSS paper

The paper's evidence comes in three buckets of runs. Each Python notebook runs
step by step and times its expensive steps; each R document is a Quarto file
(`.qmd`) that reads like a notebook: open it in RStudio and run it chunk by chunk,
or render it to HTML. All outputs go to a `results/` folder next to the file.

## Order

| Bucket | Step | File | Language | Paper | Needs |
|---|---|---|---|---|---|
| 1 Verbal aggression | 1 | `verbagg/61a_lme4_reference.qmd` | R (lme4) | 6.1, lme4 references for A1, B1, A2, B2 | — |
| | 2 | `verbagg/61_verbagg_workflow.ipynb` | Python | 6.1, workflow, agreement, probit vs logit, speed | step 1 (and optionally step 4) |
| | 3 | `capabilities/check_mirt.qmd` | R (mirt) | what mirt can and cannot fit | — |
| | 4 | `capabilities/check_brms.qmd` | R (brms, Stan) | what brms can fit; timed runs read by step 2 | a Stan toolchain |
| 2 PISA 2022 | 5 | `../notebooks/tests/50_external_crosscheck.ipynb`, Section 2 | Python | 6.2, against the published SAS estimates | the PISA extract |
| 3 Simulation | 6 | `sbc/71_sbc.ipynb` | Python | 7, simulation-based calibration | `sbc/sbc_models.py` |

Open each Python notebook from its own folder (Jupyter's working directory is
then the notebook's folder); the first code cell installs the package from this
repository in editable mode. Quarto documents run in their own folder, so no
`setwd()` is needed.

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
| 61a | lme4 A1 / B1 / A2 / B2 (one core) | run locally; written to `results/lme4_runtime_61.csv` |
| 71 | one replication: S1 / S2 / S3 | 4.6 / 6.9 / 5.2 s |

SBC at the notebook defaults (N_ITER = 8,000, 300 replications × 3 models) is
the long run: about 20 to 30 s per replication, so roughly 6 hours on one core
or about 1 hour on 7 workers. Step 3 of `71_sbc.ipynb` times one replication
per model on your machine and prints the projection before the full run; the
full run writes after every replication and resumes if interrupted.
