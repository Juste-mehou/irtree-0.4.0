# =============================================================================
# 72a  Reference fits with lme4::glmer  (paper Section 7.2, part 1)
# =============================================================================
# Run this BEFORE notebook 72_agreement.ipynb.  Open it in RStudio and run it
# section by section (Ctrl+Alt+T runs the current section), or all at once:
#   setwd("<repository root>/replication/verbagg");  source("72a_lme4_reference.R")
#
# Tree (De Boeck & Partchev, 2012): node 1 separates "no" (0) from
# "perhaps"/"yes" (1); node 2, reached only by "perhaps"/"yes", separates
# "perhaps" (0) from "yes" (1).  Each node is a logistic Rasch model with its
# own trait; the two traits are correlated.
#
#   Model A  item-by-node fixed effects (the De Boeck & Partchev model)
#   Model B  explanatory: random items within node (independent across nodes),
#            centered item covariates (behavior type, situation, mode) with a
#            node intercept, centered person covariates (anger, gender)
#
# glmer uses the Laplace approximation (nAGQ = 1; adaptive quadrature is not
# available with two correlated random effects per person).
#
# Output (results/):  lme4_modelA.csv, lme4_modelB.csv, lme4_runtime.csv,
#                     lme4_sessionInfo.txt
# Reference runtime (one core, cloud workspace): Model A ~3.3 min, B ~0.5 min.
# =============================================================================

## ---- Step 0: packages and paths --------------------------------------------
if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4")
suppressPackageStartupMessages(library(lme4))
dir.create("results", showWarnings = FALSE)
runtime <- data.frame(step = character(), seconds = numeric())
timed <- function(step, expr) {
  t0 <- proc.time()[["elapsed"]]
  val <- force(expr)
  dt <- proc.time()[["elapsed"]] - t0
  runtime[nrow(runtime) + 1, ] <<- list(step, dt)
  cat(sprintf("[%s] %.1f s\n", step, dt))
  invisible(val)
}

## ---- Step 1: read the data ------------------------------------------------
# verbagg_long.csv is lme4's VerbAgg (316 persons x 24 items), one row per
# person x item.  The same file is read by the Python notebooks.
va <- read.csv("verbagg_long.csv", stringsAsFactors = TRUE)
str(va)
table(va$resp)

## ---- Step 2: dendrify (one row per person x item x node reached) ----------
y  <- match(va$resp, c("no", "perhaps", "yes")) - 1L
n1 <- data.frame(va, node = "n1", value = as.integer(y >= 1))
n2 <- data.frame(va, node = "n2", value = as.integer(y == 2))[y >= 1, ]
long <- rbind(n1, n2)
long$node <- factor(long$node)
long$d1 <- as.numeric(long$node == "n1")   # numeric node indicators: "||"
long$d2 <- as.numeric(long$node == "n2")   # decorrelates only numeric terms
table(long$node, long$value)

## ---- Step 3: centered covariates (same coding as the Python notebooks) ----
items <- unique(va[, c("item", "btype", "situ", "mode")])
icov <- data.frame(item  = items$item,
                   scold = as.numeric(items$btype == "scold"),
                   shout = as.numeric(items$btype == "shout"),
                   self  = as.numeric(items$situ == "self"),
                   do    = as.numeric(items$mode == "do"))
for (v in c("scold", "shout", "self", "do")) icov[[v]] <- icov[[v]] - mean(icov[[v]])
pers <- unique(va[, c("id", "Anger", "Gender")])
pcov <- data.frame(id    = pers$id,
                   anger = pers$Anger - mean(pers$Anger),
                   male  = as.numeric(pers$Gender == "M") - mean(pers$Gender == "M"))
long <- merge(merge(long, icov, by = "item"), pcov, by = "id")

## ---- Step 4: helper to tidy a glmer fit ------------------------------------
tidy <- function(fit, model) {
  fe <- summary(fit)$coefficients
  out <- data.frame(model = model, term = rownames(fe), estimate = fe[, 1],
                    se = fe[, 2], row.names = NULL)
  vc <- as.data.frame(VarCorr(fit))
  vc$term <- ifelse(is.na(vc$var2), paste0("var_", vc$grp, "_", vc$var1),
                    paste0("cor_", vc$grp, "_", vc$var1, "_", vc$var2))
  vc$estimate <- ifelse(is.na(vc$var2), vc$vcov, vc$sdcor)
  rbind(out, data.frame(model = model, term = vc$term, estimate = vc$estimate,
                        se = NA_real_))
}
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

## ---- Step 5: Model A (item-by-node fixed effects) ---------------------------
fitA <- timed("lme4 Model A", glmer(value ~ 0 + item:node + (0 + node | id),
                                    data = long, family = binomial, control = ctrl))
print(VarCorr(fitA))
write.csv(tidy(fitA, "A"), "results/lme4_modelA.csv", row.names = FALSE)

## ---- Step 6: Model B (explanatory) -----------------------------------------
fitB <- timed("lme4 Model B", glmer(
  value ~ 0 + node + node:(scold + shout + self + do) + node:(anger + male) +
    (0 + node | id) + (0 + d1 + d2 || item),
  data = long, family = binomial, control = ctrl))
print(summary(fitB)$coefficients)
print(VarCorr(fitB))
write.csv(tidy(fitB, "B"), "results/lme4_modelB.csv", row.names = FALSE)

## ---- Step 7: runtime and session -------------------------------------------
print(runtime)
write.csv(runtime, "results/lme4_runtime.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "results/lme4_sessionInfo.txt")
