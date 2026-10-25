# Section 7.2: reference maximum-likelihood fits of the verbal aggression
# linear tree with lme4::glmer (Laplace approximation).
#
# Tree (De Boeck & Partchev, 2012): node 1 separates "no" (0) from
# "perhaps"/"yes" (1); node 2, reached only by "perhaps"/"yes", separates
# "perhaps" (0) from "yes" (1).  Each node is a logistic Rasch model with its
# own trait; the two traits are correlated.
#
# Model A: item-by-node fixed effects (the De Boeck & Partchev model).
# Model B: explanatory.  Items are random within node (independent across
#          nodes), with item covariates (btype, situ, mode) and person
#          covariates (Anger, Gender), all centered, and a node intercept.
#
# Usage:  Rscript replication/verbagg/fit_lme4.R   (from the repository root)
# Output: replication/verbagg/results/lme4_modelA.csv, lme4_modelB.csv

suppressPackageStartupMessages(library(lme4))
here <- "replication/verbagg"
dir.create(file.path(here, "results"), showWarnings = FALSE)
va <- read.csv(file.path(here, "verbagg_long.csv"), stringsAsFactors = TRUE)

# --- dendrify: one row per person x item x node that the response reaches ---
y <- match(va$resp, c("no", "perhaps", "yes")) - 1L
n1 <- data.frame(va, node = "n1", value = as.integer(y >= 1))
n2 <- data.frame(va, node = "n2", value = as.integer(y == 2))[y >= 1, ]
long <- rbind(n1, n2)
long$node <- factor(long$node)
long$d1 <- as.numeric(long$node == "n1")   # numeric node indicators: "||" only
long$d2 <- as.numeric(long$node == "n2")   # decorrelates numeric terms

# --- centered covariates (the same coding is used by the irtree scripts) ---
items <- unique(va[, c("item", "btype", "situ", "mode")])
icov <- data.frame(item   = items$item,
                   scold  = as.numeric(items$btype == "scold"),
                   shout  = as.numeric(items$btype == "shout"),
                   self   = as.numeric(items$situ == "self"),
                   do     = as.numeric(items$mode == "do"))
for (v in c("scold", "shout", "self", "do")) icov[[v]] <- icov[[v]] - mean(icov[[v]])
pers <- unique(va[, c("id", "Anger", "Gender")])
pcov <- data.frame(id = pers$id,
                   anger = pers$Anger - mean(pers$Anger),
                   male  = as.numeric(pers$Gender == "M") - mean(pers$Gender == "M"))
long <- merge(merge(long, icov, by = "item"), pcov, by = "id")

ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

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

t0 <- Sys.time()
fitA <- glmer(value ~ 0 + item:node + (0 + node | id), data = long,
              family = binomial, control = ctrl)
cat("Model A:", format(Sys.time() - t0), " logLik", logLik(fitA), "\n")
write.csv(tidy(fitA, "A"), file.path(here, "results", "lme4_modelA.csv"), row.names = FALSE)

t0 <- Sys.time()
fitB <- glmer(value ~ 0 + node + node:(scold + shout + self + do) +
                node:(anger + male) + (0 + node | id) + (0 + d1 + d2 || item),
              data = long, family = binomial, control = ctrl)
cat("Model B:", format(Sys.time() - t0), " logLik", logLik(fitB), "\n")
write.csv(tidy(fitB, "B"), file.path(here, "results", "lme4_modelB.csv"), row.names = FALSE)

writeLines(capture.output(sessionInfo()), file.path(here, "results", "lme4_sessionInfo.txt"))
