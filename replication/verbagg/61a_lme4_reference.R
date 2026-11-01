# =============================================================================
# 61a  lme4 reference fits for bucket 1 (paper Section 6.1, notebook 61)
# =============================================================================
# Run this BEFORE the comparison steps of 61_verbagg_workflow.ipynb.  In RStudio:
#   setwd("<repository root>/replication/verbagg");  source("61a_lme4_reference.R")
# (run section by section with Ctrl+Alt+T, or all at once).
#
# The four irtree models of bucket 1 and their lme4 counterparts:
#
#   irtree  link    covariates  lme4 model (glmer, Laplace approximation)
#   A1      logit   none        items fixed by node: value ~ 0 + item:node + (0 + node | id)
#   B1      probit  none        the same, binomial(link = "probit")
#   A2      logit   yes         node intercepts + covariates, items random by node:
#                               value ~ 0 + node + node:(scold + shout + self + do)
#                                       + node:(anger + male) + (0 + node | id) + (0 + d1 + d2 || item)
#   B2      probit  yes         the same, binomial(link = "probit")
#
# Covariate coding, identical to the notebook: anger centered and scaled over
# persons; male and the item indicators (scold, shout, self, do) centered.
#
# Output (results/): lme4_A1.csv, lme4_B1.csv, lme4_A2.csv, lme4_B2.csv,
#                    lme4_runtime_61.csv, lme4_sessionInfo_61.txt
# Reference runtime (one core): A1 and B1 about 1.5 to 3 min each, A2 and B2
# about 0.5 min each.
# =============================================================================

## ---- Step 0: packages and paths --------------------------------------------
if (!requireNamespace("lme4", quietly = TRUE)) install.packages("lme4")
suppressPackageStartupMessages(library(lme4))
dir.create("results", showWarnings = FALSE)
runtime <- data.frame(model = character(), seconds = numeric())
timed <- function(label, expr) {
  t0 <- proc.time()[["elapsed"]]
  val <- force(expr)
  dt <- proc.time()[["elapsed"]] - t0
  runtime[nrow(runtime) + 1, ] <<- list(label, dt)
  cat(sprintf("[%s] %.1f s\n", label, dt))
  invisible(val)
}

## ---- Step 1: read and dendrify ------------------------------------------------
va <- read.csv("verbagg_long.csv", stringsAsFactors = TRUE)
y  <- match(va$resp, c("no", "perhaps", "yes")) - 1L
n1 <- data.frame(va, node = "n1", value = as.integer(y >= 1))
n2 <- data.frame(va, node = "n2", value = as.integer(y == 2))[y >= 1, ]
long <- rbind(n1, n2)
long$node <- factor(long$node)
long$d1 <- as.numeric(long$node == "n1")   # numeric node indicators: "||" only
long$d2 <- as.numeric(long$node == "n2")   # decorrelates numeric terms

## ---- Step 2: covariates (same coding as the notebook) -------------------------
items <- unique(va[, c("item", "btype", "situ", "mode")])
icov <- data.frame(item  = items$item,
                   scold = as.numeric(items$btype == "scold"),
                   shout = as.numeric(items$btype == "shout"),
                   self  = as.numeric(items$situ == "self"),
                   do    = as.numeric(items$mode == "do"))
for (v in c("scold", "shout", "self", "do")) icov[[v]] <- icov[[v]] - mean(icov[[v]])
pers <- unique(va[, c("id", "Anger", "Gender")])
pcov <- data.frame(id    = pers$id,
                   anger = (pers$Anger - mean(pers$Anger)) / sd(pers$Anger),
                   male  = as.numeric(pers$Gender == "M") - mean(pers$Gender == "M"))
long <- merge(merge(long, icov, by = "item"), pcov, by = "id")

## ---- Step 3: helpers ------------------------------------------------------------
tidy <- function(fit, model) {
  fe <- summary(fit)$coefficients
  out <- data.frame(model = model, term = rownames(fe), estimate = fe[, 1],
                    se = fe[, 2], row.names = NULL)
  vc <- as.data.frame(VarCorr(fit))
  vc$term <- ifelse(is.na(vc$var2), paste0("var_", vc$grp, "_", vc$var1),
                    paste0("cor_", vc$grp, "_", vc$var1, "_", vc$var2))
  vc$estimate <- ifelse(is.na(vc$var2), vc$vcov, vc$sdcor)
  rbind(out, data.frame(model = model, term = vc$term, estimate = vc$estimate, se = NA_real_))
}
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
f_fixed <- value ~ 0 + item:node + (0 + node | id)
f_cov   <- value ~ 0 + node + node:(scold + shout + self + do) + node:(anger + male) +
             (0 + node | id) + (0 + d1 + d2 || item)

## ---- Step 4: A1 and B1 (items fixed, no covariates) ------------------------------
fit_A1 <- timed("lme4 A1 (logit, items fixed)",
                glmer(f_fixed, data = long, family = binomial, control = ctrl))
write.csv(tidy(fit_A1, "A1"), "results/lme4_A1.csv", row.names = FALSE)
fit_B1 <- timed("lme4 B1 (probit, items fixed)",
                glmer(f_fixed, data = long, family = binomial(link = "probit"), control = ctrl))
write.csv(tidy(fit_B1, "B1"), "results/lme4_B1.csv", row.names = FALSE)
print(VarCorr(fit_A1)); print(VarCorr(fit_B1))

## ---- Step 5: A2 and B2 (random items, intercepts, covariates) ---------------------
fit_A2 <- timed("lme4 A2 (logit, covariates)",
                glmer(f_cov, data = long, family = binomial, control = ctrl))
write.csv(tidy(fit_A2, "A2"), "results/lme4_A2.csv", row.names = FALSE)
fit_B2 <- timed("lme4 B2 (probit, covariates)",
                glmer(f_cov, data = long, family = binomial(link = "probit"), control = ctrl))
write.csv(tidy(fit_B2, "B2"), "results/lme4_B2.csv", row.names = FALSE)
print(summary(fit_A2)$coefficients); print(VarCorr(fit_A2))

## ---- Step 6: runtime and session ----------------------------------------------------
print(runtime)
write.csv(runtime, "results/lme4_runtime_61.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "results/lme4_sessionInfo_61.txt")
