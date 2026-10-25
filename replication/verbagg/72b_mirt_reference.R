# =============================================================================
# 72b  Reference fits with mirt  (paper Section 7.2, part 2)
# =============================================================================
# Run this BEFORE part 2 of notebook 72_agreement.ipynb.  In RStudio:
#   setwd("<repository root>/replication/verbagg");  source("72b_mirt_reference.R")
#
# Why mirt: an IRTree is a multidimensional IRT model with simple structure
# fitted to the pseudo-item responses, cells a path does not reach being
# missing (Jeon & De Boeck, 2016).  mirt fits that model by marginal maximum
# likelihood with fixed quadrature (EM), so it covers what glmer cannot: free
# item slopes, partial credit items, and correlated traits with item slopes.
# mirt fixes the latent means at 0 and variances at 1, the same convention as
# irtree on dimensions whose nodes all have slopes.
#
#   Model M1  linear tree, logistic 2PL at both nodes, correlated traits
#             (48 pseudo-items: 24 for node 1, 24 for node 2 with NA where
#             node 2 is not reached)
#   Model M2  one-node tree, generalized partial credit model on the three
#             ordered categories (no / perhaps / yes)
#
# mirt's 2PL is  logit P = a*theta + d ; irtree's is a*(theta - b), so the
# notebook compares a and d = -a*b (computed draw by draw).  mirt's GPCM with
# IRTpars = TRUE gives a and Muraki step parameters b1, b2, the irtree form.
#
# Output (results/): mirt_M1_items.csv, mirt_M1_cor.csv, mirt_M2_items.csv,
#                    mirt_runtime.csv, mirt_sessionInfo.txt
# =============================================================================

## ---- Step 0: packages and paths --------------------------------------------
if (!requireNamespace("mirt", quietly = TRUE)) install.packages("mirt")
suppressPackageStartupMessages(library(mirt))
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

## ---- Step 1: read the data and make the wide response matrix ---------------
va <- read.csv("verbagg_long.csv", stringsAsFactors = TRUE)
va$y <- match(va$resp, c("no", "perhaps", "yes")) - 1L
items <- unique(as.character(va$item))            # lme4 order, as in Python
persons <- sort(unique(va$id))
Y <- matrix(NA_integer_, length(persons), length(items),
            dimnames = list(persons, items))
Y[cbind(match(va$id, persons), match(va$item, items))] <- va$y
table(Y)

## ---- Step 2: pseudo-items of the linear tree --------------------------------
N1 <- (Y >= 1) * 1L                              # node 1: no (0) vs perhaps/yes (1)
N2 <- ifelse(Y >= 1, (Y == 2) * 1L, NA_integer_) # node 2: perhaps (0) vs yes (1)
colnames(N1) <- paste0(items, "_n1")
colnames(N2) <- paste0(items, "_n2")
pseudo <- cbind(N1, N2)
colSums(!is.na(pseudo))[c(1, 25)]                # cells reaching each node

## ---- Step 3: Model M1, two-dimensional 2PL tree with correlated traits -----
spec_M1 <- mirt.model("F1 = 1-24
                       F2 = 25-48
                       COV = F1*F2")
fitM1 <- timed("mirt M1 (2PL tree)",
               mirt(pseudo, spec_M1, itemtype = "2PL", method = "EM",
                    SE = TRUE, verbose = FALSE))
print(fitM1)
cf <- coef(fitM1, printSE = TRUE)
rows <- lapply(seq_along(colnames(pseudo)), function(j) {
  m <- cf[[j]]
  node <- if (j <= 24) 1L else 2L
  a_col <- if (node == 1L) "a1" else "a2"
  data.frame(item = items[(j - 1) %% 24 + 1], item_index = (j - 1) %% 24 + 1,
             node = node,
             a = m["par", a_col], a_se = m["SE", a_col],
             d = m["par", "d"],   d_se = m["SE", "d"])
})
M1 <- do.call(rbind, rows)
write.csv(M1, "results/mirt_M1_items.csv", row.names = FALSE)
gp <- cf$GroupPars
cor_M1 <- data.frame(estimate = gp["par", "COV_21"], se = gp["SE", "COV_21"])
print(cor_M1)                                    # variances are fixed at 1
write.csv(cor_M1, "results/mirt_M1_cor.csv", row.names = FALSE)

## ---- Step 4: Model M2, one-node GPCM ---------------------------------------
fitM2 <- timed("mirt M2 (GPCM)",
               mirt(as.data.frame(Y), 1, itemtype = "gpcm", SE = TRUE, verbose = FALSE))
cf2 <- coef(fitM2, IRTpars = TRUE, printSE = TRUE)
M2 <- do.call(rbind, lapply(seq_along(items), function(j) {
  m <- cf2[[j]]
  data.frame(item = items[j], item_index = j,
             a = m["par", "a"], a_se = m["SE", "a"],
             b1 = m["par", "b1"], b1_se = m["SE", "b1"],
             b2 = m["par", "b2"], b2_se = m["SE", "b2"])
}))
write.csv(M2, "results/mirt_M2_items.csv", row.names = FALSE)
head(M2)

## ---- Step 5: runtime and session -------------------------------------------
print(runtime)
write.csv(runtime, "results/mirt_runtime.csv", row.names = FALSE)
writeLines(capture.output(sessionInfo()), "results/mirt_sessionInfo.txt")
