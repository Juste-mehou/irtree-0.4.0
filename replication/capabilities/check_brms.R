# =============================================================================
# What can brms do for IRTree models?  A capability and speed check on the
# verbal aggression linear tree (Buerkner, 2021, JSS 100(5), for the syntax).
# =============================================================================
# Part 1 tries each feature with a short run (1 chain, few iterations): does it
# compile and sample?  Part 2 times full runs of the models irtree is compared
# with in bucket 1 (Rasch tree without and with covariates, and a 2PL tree).
# Run from replication/capabilities:
#   setwd("<repository root>/replication/capabilities"); source("check_brms.R")
# Needs a working Stan toolchain (rstan, or cmdstanr with CmdStan installed).
# Output: results/capabilities_brms.csv, results/brms_speed.csv,
#         results/brms_estimates.csv, results/capabilities_brms_sessionInfo.txt
# Reference: on a laptop, Part 1 takes about 10 to 20 min (each model
# compiles once, about 30 to 60 s); Part 2 about 20 to 60 min.
# =============================================================================

## ---- settings ----------------------------------------------------------------
RUN_PART2   <- TRUE     # full timed runs
FULL_ITER   <- 2000     # per chain, half warm-up (the brms default)
FULL_CHAINS <- 4
for (p in c("brms", "posterior", "BH")) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
suppressPackageStartupMessages({ library(brms); library(posterior) })
BACKEND <- if (requireNamespace("cmdstanr", quietly = TRUE) &&
               !is.null(tryCatch(cmdstanr::cmdstan_version(), error = function(e) NULL))) "cmdstanr" else "rstan"
cat("backend:", BACKEND, "\n")
options(mc.cores = FULL_CHAINS)
dir.create("results", showWarnings = FALSE)

## ---- data: dendrified long format (as in 72a_lme4_reference.R) ---------------
va <- read.csv("../verbagg/verbagg_long.csv", stringsAsFactors = TRUE)
y  <- match(va$resp, c("no", "perhaps", "yes")) - 1L
n1 <- data.frame(va, y3 = y, node = "n1", value = as.integer(y >= 1))
n2 <- data.frame(va, y3 = y, node = "n2", value = as.integer(y == 2))[y >= 1, ]
long <- rbind(n1, n2)
long$node <- factor(long$node)
long$item_node <- interaction(long$item, long$node, drop = TRUE)
long$anger <- as.numeric(scale(long$Anger))           # standardized (scale per person below)
pa <- unique(va[, c("id", "Anger")]); m <- mean(pa$Anger); s <- sd(pa$Anger)
long$anger <- (long$Anger - m) / s                     # centered and scaled over persons
long$male  <- as.numeric(long$Gender == "M") - mean(unique(va[, c("id", "Gender")])$Gender == "M")
for (v in c("scold", "shout")) long[[v]] <- as.numeric(long$btype == v)
long$self <- as.numeric(long$situ == "self"); long$do <- as.numeric(long$mode == "do")
it <- unique(long[, c("item", "scold", "shout", "self", "do")])
for (v in c("scold", "shout", "self", "do")) long[[v]] <- long[[v]] - mean(it[[v]])

## ---- harness ---------------------------------------------------------------------
results <- data.frame(test = character(), status = character(), seconds = numeric(),
                      note = character(), stringsAsFactors = FALSE)
check <- function(test, expr, note = "") {
  t0 <- proc.time()[["elapsed"]]
  status <- tryCatch({ force(expr); "ok" },
                     error = function(e) paste("error:", conditionMessage(e)))
  dt <- proc.time()[["elapsed"]] - t0
  results[nrow(results) + 1, ] <<- list(test, substr(status, 1, 300), round(dt, 1), note)
  cat(sprintf("[%-48s] %6.1f s  %s\n", test, dt, substr(status, 1, 120)))
  invisible(status)
}
quick <- function(formula, data = long, family = bernoulli(), prior = NULL, ...) {
  args <- list(formula, data = data, prior = prior, chains = 1, iter = 300, warmup = 150,
               refresh = 0, backend = BACKEND, silent = 2, ...)
  if (!is.null(family)) args$family <- family        # NULL: families set inside bf()
  do.call(brm, args)
}

## ---- formulas -------------------------------------------------------------------------
f_rasch_fixed <- bf(value ~ 0 + item_node + (0 + node | id))
f_rasch_rand  <- bf(value ~ 0 + node + (0 + node | id) + (0 + node | item))
f_rasch_cov   <- bf(value ~ 0 + node + node:(scold + shout + self + do) + node:(anger + male) +
                      (0 + node | id) + (0 + node | item))
f_2pl <- bf(value ~ exp(logalpha) * eta,
            eta ~ 0 + node + (0 + node | i | item) + (0 + node | id),
            logalpha ~ 0 + node + (0 + node | i | item), nl = TRUE)
p_2pl <- prior(constant(1), class = "sd", group = "id", nlpar = "eta") +
         prior(normal(0, 1), nlpar = "logalpha") + prior(normal(0, 3), nlpar = "eta")
f_3pl <- bf(value ~ gamma + (1 - gamma) * inv_logit(exp(logalpha) * eta),
            eta ~ 0 + node + (0 + node | i | item) + (0 + node | id),
            logalpha ~ 0 + node + (0 + node | i | item),
            nlf(gamma ~ inv_logit(logitgamma)), logitgamma ~ 1, nl = TRUE)
p_3pl <- p_2pl + prior(normal(-1.5, 0.5), nlpar = "logitgamma")

## ---- Part 1: capabilities (short runs) ---------------------------------------------------
check("Rasch tree, logit, items fixed (De Boeck & Partchev)", quick(f_rasch_fixed))
check("Rasch tree, logit, items random per node", quick(f_rasch_rand))
check("Rasch tree, probit", quick(f_rasch_rand, family = bernoulli(link = "probit")))
check("Rasch tree with item and person covariates", quick(f_rasch_cov))
check("2PL tree (non-linear formula), slopes per item and node", quick(f_2pl, prior = p_2pl))
check("2PL tree, probit", quick(f_2pl, prior = p_2pl, family = bernoulli(link = "probit")))
check("2PL tree with person covariates on eta",
      quick(bf(value ~ exp(logalpha) * eta,
               eta ~ 0 + node + node:(anger + male) + (0 + node | i | item) + (0 + node | id),
               logalpha ~ 0 + node + (0 + node | i | item), nl = TRUE), prior = p_2pl))
check("3PL node (lower asymptote via inv_logit)",
      quick(f_3pl, prior = p_3pl, family = bernoulli(link = "identity")))
one <- unique(long[, c("id", "item", "y3")])
one$y3 <- one$y3 + 1L                                   # cumulative / acat need categories 1, 2, 3
check("graded response, one node (cumulative logit)",
      quick(bf(y3 ~ 1 + (1 | item) + (1 | id)), data = one, family = cumulative("logit")))
check("partial credit / adjacent category, one node",
      quick(bf(y3 ~ 1 + (1 | item) + (1 | id)), data = one, family = acat("logit")))
check("mixed node (binary items and three-category items in one model)",
      stop("not expressible with one family; would need a custom family or two linked models"),
      note = "recorded as not available unless a custom family is written")
long_na <- long; long_na$anger[long_na$id %in% c(3, 17, 40)] <- NA
check("missing person covariate (mi() imputation)",
      quick(bf(value ~ 0 + node + node:mi(anger) + (0 + node | id) + (0 + node | item),
               family = bernoulli()) +
              bf(anger | mi() ~ 1, family = gaussian()) + set_rescor(FALSE),
            data = long_na, family = NULL),
      note = "mi() imputes each ROW separately: a person's anger is not held equal across rows")

print(results, right = FALSE)
write.csv(results, "results/capabilities_brms.csv", row.names = FALSE)

## ---- Part 2: timed full runs ------------------------------------------------------------------
if (RUN_PART2) {
  timed_fit <- function(label, formula, prior = NULL, family = bernoulli()) {
    t0 <- proc.time()[["elapsed"]]
    fit <- brm(formula, data = long, family = family, prior = prior, chains = FULL_CHAINS,
               iter = FULL_ITER, cores = FULL_CHAINS, refresh = 0, backend = BACKEND, silent = 2)
    total <- proc.time()[["elapsed"]] - t0
    el <- tryCatch(rstan::get_elapsed_time(fit$fit), error = function(e) NULL)
    sampling <- if (is.null(el)) NA else max(rowSums(el))      # slowest chain, parallel chains
    dr <- summarise_draws(as_draws_df(fit), "ess_bulk")
    keep <- grepl("^(sd_|cor_|b_)", dr$variable)
    list(fit = fit, row = data.frame(model = label, chains = FULL_CHAINS, iter = FULL_ITER,
                                     total_seconds = round(total, 1),
                                     sampling_seconds = round(sampling, 1),
                                     min_ess_bulk_structural = round(min(dr$ess_bulk[keep], na.rm = TRUE)),
                                     backend = BACKEND))
  }
  runs <- list(
    timed_fit("brms Rasch tree, items fixed (A1)", f_rasch_fixed),
    timed_fit("brms Rasch tree, random items + covariates (A2)", f_rasch_cov),
    timed_fit("brms Rasch tree, probit, random items + covariates (B2)", f_rasch_cov,
              family = bernoulli(link = "probit")),
    timed_fit("brms 2PL tree", f_2pl, prior = p_2pl))
  speed <- do.call(rbind, lapply(runs, `[[`, "row"))
  print(speed)
  write.csv(speed, "results/brms_speed.csv", row.names = FALSE)
  est <- do.call(rbind, lapply(runs, function(r) {
    s <- summarise_draws(as_draws_df(r$fit), "mean", "sd")
    s <- s[grepl("^(sd_|cor_|b_)", s$variable), ]
    data.frame(model = r$row$model, s)
  }))
  write.csv(est, "results/brms_estimates.csv", row.names = FALSE)
}
writeLines(capture.output(sessionInfo()), "results/capabilities_brms_sessionInfo.txt")
