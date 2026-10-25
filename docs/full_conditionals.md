# Full Conditional Derivations for the `irtree` Package

Derivations of the full conditional distributions used by the Gibbs and MH updates in the `irtree` sampler.

> **Scope (v5, package 0.4).** Reconciled against the code after version 0.3.0.
> Changes from v4:
>
> - **Identification defaults** stated up front (§1.4): θ has mean 0 everywhere and
>   variance 1 on dimensions with free slopes; Rasch-type dimensions estimate it.
> - **One latent view** for every augmented node (§1.5); θ and the coefficients are
>   built from it, not from `compute_suff_stats` (kept only for compatibility).
> - **GRM thresholds** (§2.10) corrected: given Z the full conditional is the *prior*
>   truncated to the interval the categories allow, with no Normal likelihood term;
>   each iteration first runs a Cowles step with Z integrated out (§5.1).
> - **Discrimination** (§3.1, §3.1a): the truncated-Normal prior is the default
>   (Gibbs); LogNormal is opt-in (MH, target written on ξ = log a). One slope shared by
>   all items of a node (`common_discrimination`) pools every cell (§3.1a).
> - **Covariates** (§2.11): one function, `gibbs_coef_pooled`, for β, κ and γ, with each
>   node's own discriminations; missing X, Q, W drawn feature by feature through one
>   Gaussian algebra (§2.11.2). κ is identified through b's prior (§2.11.5).
> - **Correlated item parameters** (§2.12–2.13): pooled over every node of each group;
>   a-correlation groups use componentwise exact Gibbs (no rejection step).
> - **θ correlation blocks** (§2.6, §3.3), Inverse-Wishart per block (§2.9, §2.14).
> - **New:** non-augmented nodes (logit 2PL/3PL, GPCM/PCM, custom) and groups that mix
>   probit and logit members (§3.4–3.5); moves that improve mixing (§5: Cowles step,
>   location move, interweaving); the logistic-equivalent rescaling (§1.1).

**Notation used throughout:**

- $p = 1, \ldots, P$ persons. $i = 1, \ldots, I$ items. $r = 1, \ldots, R$ nodes. $d = 1, \ldots, D$ latent dimensions ($D \le R$ when nodes share a trait).
- $Y_{pi}$ is the observed category. $Y^{*}_{pir}$ is the pseudo-response at node $r$.
- $T_{r}[p, i] = 1$ if node $r$ is on the path to $Y_{pi}$, else $0$.
- $\theta_{pr}$ is the trait of person $p$ at node $r$ (the trait of node $r$'s dimension).
- $b_{ir}$ is the difficulty of item $i$ at node $r$; $\tau_{i,k}$ are GRM thresholds.
- $a_{ir}$, $c_{ir}$ are discrimination and guessing parameters where present ($a_{ir}=1$ at Rasch and GRM nodes).
- $\text{offset}_{pir} = X_p^\top\beta + Q_i^\top\kappa + W_{pi}^\top\gamma$ is the covariate part of the linear predictor (zero without covariates).
- $\sigma^{2}_{r}$ is the variance of the person-trait prior at node $r$.
- $\mu_b$, $\tau^{2}$ are the mean and variance of the item-difficulty prior ($\mu_b = 0$ by default).
- $\Phi$ is the standard normal CDF. $\phi$ is the standard normal PDF.
- $\propto$ means proportional to as a function of the parameter being updated.

------------------------------------------------------------------------

## 1. Preliminaries

### 1.1 What Albert-Chib augmentation actually does

Start with one binary outcome. It is Bernoulli:

$$\Pr(Y^{*}_{pir} = y) = p^{\,y}(1-p)^{1-y}.$$

Here $p$ is one number. The same for everyone. That is not enough. We want $p$ to change across persons and items. We want it to carry a person parameter and an item parameter.

So we make $p$ a function. We write $p_{pir}$ and build it from $\theta_{pr}$ and $b_{ir}$:

$$\Pr(Y^{*}_{pir} = y \mid p_{pir}) = p_{pir}^{\,y}(1-p_{pir})^{1-y}.$$

Now we need $p_{pir}$ to live between 0 and 1. We get it from a latent variable.

Posit a continuous variable $Y_{pir}$:

$$Y_{pir} = (\theta_{pr} - b_{ir}) + \epsilon_{pir}.$$

The observed response is just which side of zero this lands on:

$$Y^{*}_{pir} = \mathbb{1}\{Y_{pir} > 0\}.$$

So $p_{pir} = \Pr(Y_{pir} > 0 \mid \theta_{pr}, b_{ir})$. The distribution of $\epsilon_{pir}$ fixes the form of $p_{pir}$ and therefore the values of $\theta_{pr}$ and $b_{ir}$ drawn at each iteration:

- $\epsilon \sim \mathcal{N}(0,1)$ gives $p_{pir} = \Phi(\theta_{pr} - b_{ir})$. This is probit.
- $\epsilon \sim \text{Logistic}(0,1)$ gives $p_{pir} = g(\theta_{pr} - b_{ir})$. This is logit.

The link function is not a separate choice. It is the CDF of $\epsilon$.

**A note on the 1.702 approximation.** The probit and logit ICCs are nearly identical numerically: $\Phi(x)$ is within 0.0095 of $g(1.702\,x)$ for every $x$, and no other single constant does better (Haley, 1952). This is why probit and logit yield the same substantive conclusions in practice: the models are nearly interchangeable up to a scale factor. We choose probit not because it fits data better but because $\epsilon \sim \mathcal{N}(0,1)$ enables conjugacy, as explained below.

`fit.to_logit()` uses this to report a probit fit on the logistic-equivalent scale. The factor 1.702 multiplies the linear predictor $a(\theta - b + \text{offset})$, so it goes on whichever quantity carries the latent scale: on a dimension whose nodes are all probit without slopes, $\theta$, $b$, thresholds and covariate effects are multiplied by 1.702 (variances by $1.702^2$); elsewhere the probit slopes are. Correlations and guessing parameters are unchanged. The match is weakest in the tails, where the logistic is heavier: for nodes with rare outcomes (rapid guessing, say) a logit refit can differ from the rescaled probit fit by 5–15%.

Now the trick. Albert-Chib adds $Y_{pir}$ to the list of things we sample. Before, our unknowns were $\theta$, $b$, $\sigma^2$. After augmentation, our unknowns are $\theta$, $b$, $\sigma^2$, and $Y$. That is all augmentation does. We sample one more thing.

Why does this help? Once we know $Y_{pir}$, the observed $Y^{*}_{pir}$ tells us nothing new. $Y^{*}$ is just the sign of $Y$. It is fixed the moment $Y$ is drawn.

So in any full conditional, the term $\Pr(Y^{*}_{pir} \mid \cdots)$ drops out entirely. It does not depend on $\theta$ anymore, or on $b$, or on any parameter we want to update. What is left is the density of $Y_{pir}$ times the prior on the parameter.

For probit, the density of $Y_{pir}$ given $\theta, b$ is Normal. The prior on $\theta$ is Normal. Normal times Normal is Normal. Closed form. No accept-reject.

**This applies to every parameter whose prior is conjugate to a Normal likelihood.** Not just $\theta$. Once $Y$ is drawn, the full conditional of $b_{ir}$ also becomes Normal-times-Normal. Same for $\sigma^2$ with an InvGamma prior. The augmentation unlocks Gibbs for all of them at once, not just for $\theta$.

One caution. The distribution of $\epsilon$ is not a free choice. It is fixed by the model. If we declared a probit model, $\epsilon$ is Normal. If we declared a logit model, $\epsilon$ is Logistic. Forcing $\epsilon$ to be Normal when we wanted logit quietly switches us to a different model. For logit, Normal augmentation does not apply. A different augmentation scheme (Pólya-Gamma) is needed there.

------------------------------------------------------------------------

### 1.2 The Normal–Normal completing-the-square template

Every conjugate $\theta$ and $b$ derivation below uses this structure. We establish it once.

**Setup.** Let $\theta$ be a scalar with prior $\theta \sim \mathcal{N}(\mu_{0}, v_{0})$. We observe $n$ independent data points $w_{1}, \ldots, w_{n}$ with $w_{j} \mid \theta \sim \mathcal{N}(\theta, v)$.

**Step 1.** Write the log posterior up to a constant:

$$\log p(\theta \mid w) = -\frac{(\theta - \mu_{0})^{2}}{2v_{0}} - \frac{1}{2v}\sum_{j}(w_{j} - \theta)^{2} + \text{const}.$$

**Step 2.** Expand the squares. Drop constants in $\theta$:

$$= -\frac{1}{2}\left(\frac{1}{v_{0}} + \frac{n}{v}\right)\theta^{2} + \left(\frac{\mu_{0}}{v_{0}} + \frac{\sum_{j} w_{j}}{v}\right)\theta + \text{const}.$$

**Step 3.** A log-density quadratic in $\theta$ is always Normal. Read off the parameters:

$$\boxed{
\theta \mid w \sim \mathcal{N}(\mu_{*}, v_{*})
\quad\text{with}\quad
v_{*} = \left(\frac{1}{v_{0}} + \frac{n}{v}\right)^{-1},
\quad
\mu_{*} = v_{*}\left(\frac{\mu_{0}}{v_{0}} + \frac{\sum_{j} w_{j}}{v}\right).
}$$

**Interpretation.** Posterior precision = prior precision + data precision. Posterior mean = precision-weighted average of prior mean and data mean. Every conjugate derivation below is just identifying what plays the role of $w_{j}$, $v$, $\mu_{0}$, and $v_{0}$.

------------------------------------------------------------------------

### 1.3 The Normal–Inverse-Gamma template

When the parameter is a variance, the conjugate prior is Inverse-Gamma.

**Setup.** Let $\sigma^{2} \sim \mathrm{InvGamma}(\alpha_{0}, \beta_{0})$, so $p(\sigma^{2}) \propto (\sigma^{2})^{-\alpha_{0}-1}\exp(-\beta_{0}/\sigma^{2})$. Observe $\theta_{1}, \ldots, \theta_{P}$ with $\theta_{p} \mid \sigma^{2} \sim \mathcal{N}(0, \sigma^{2})$.

Multiply prior by likelihood and collect:

$$p(\sigma^{2} \mid \theta) \propto (\sigma^{2})^{-(\alpha_{0} + P/2) - 1} \exp\!\left(-\frac{1}{\sigma^{2}}\left[\beta_{0} + \tfrac{1}{2}\sum_{p}\theta_{p}^{2}\right]\right).$$

That is the kernel of an InvGamma. Read off:

$$\boxed{\sigma^{2} \mid \theta \sim \mathrm{InvGamma}\!\left(\alpha_{0} + \frac{P}{2},\; \beta_{0} + \frac{1}{2}\sum_{p=1}^{P}\theta_{p}^{2}\right).}$$

------------------------------------------------------------------------

### 1.4 Identification defaults

Two constraints fix the latent scale, and every derivation below assumes them.

- **Location.** Every trait has prior mean 0. Shifting all $\theta_{pr}$ and all $b_{ir}$ of a dimension by the same constant leaves every likelihood term unchanged, so the zero mean of $\theta$ is what pins the location. (The θ prior mean is therefore not a free setting.)
- **Scale.** On a dimension whose nodes all have free discriminations, the likelihood depends on $a_{ir}\theta_{pr}$ only through the product, so $a \to c\,a$, $\theta \to \theta/c$ leaves it unchanged. `Priors.from_spec` fixes $\sigma^2_d = 1$ on those dimensions (the usual 2PL convention). On a dimension with at least one node with $a = 1$ (Rasch, GRM, PCM), the scale is fixed by that node and $\sigma^2_d$ is estimated (§2.7); fixing it as well would be a second, substantive restriction. The transformation rescales each dimension separately, and correlations are scale-free, so the correlations of $\theta$ are identified under either convention. A user `priors.set("theta", ...)` replaces the default.

### 1.5 One latent view for every augmented node

Every probit node (Rasch, 2PL, 3PL, GRM, mixed-format GRM) is augmented. On the cells where $Z$ is drawn (the node's *latent mask*: every relevant cell, or the $U_{pir}=0$ ability cells at a 3PL node),

$$Z_{pir} = a_{ir}\bigl(\text{core}_{pir} + \text{offset}_{pir}\bigr) + \epsilon_{pir}, \qquad \epsilon_{pir} \sim \mathcal{N}(0,1),$$

with $\text{core} = \theta - b$ at binary nodes, $\text{core} = \theta$ at GRM nodes (the thresholds carry the location), and $\theta - b$ on the binary items of a mixed node. Every Gibbs statistic in §2 is built from this one statement (`_latent_view` / `_latent_resid` in `run_chain`): to update a block, subtract from $Z$ everything that does not involve it, and what is left is a Gaussian regression on the block. Masks, discriminations, offsets and node types are handled in one place, so the formulas below are written for one node and pool over nodes by adding precisions and linear statistics.

## 2. Conjugate parameters (Z-augmented Gibbs)

Throughout this section, $Z_{pir}$ means the augmented latent variable $Y_{pir}$ drawn for the relevant $(p,i)$ pair at node $r$. We use $Z$ to follow the standard Albert-Chib notation.

### 2.1 $\theta_{pr}$ — binary probit Rasch (1PL) node

The augmented model is:

$$Z_{pir} \mid \theta_{pr}, b_{ir} \sim \mathcal{N}(\theta_{pr} - b_{ir},\; 1), \qquad \theta_{pr} \sim \mathcal{N}(0, \sigma^{2}_{r}).$$

We want $p(\theta_{pr} \mid Z_{r}, b_{r}, \sigma^{2}_{r})$.

**Step 1.** Which $Z$ values depend on $\theta_{pr}$? Only those at items relevant to person $p$ at node $r$: $\mathcal{I}_{pr} = \{i : T_r[p,i] = 1\}$. Let $n_{pr} = |\mathcal{I}_{pr}|$.

**Step 2.** Define $w_{i} = Z_{pir} + b_{ir}$. Then $w_{i} = \theta_{pr} + \epsilon$ with $\epsilon \sim \mathcal{N}(0,1)$, so $w_{i} \mid \theta_{pr} \sim \mathcal{N}(\theta_{pr}, 1)$. That is the template with $\mu_{0} = 0$, $v_{0} = \sigma^{2}_{r}$, $v = 1$, $n = n_{pr}$.

**Step 3.** Apply:

$$v_{*} = \left(\frac{1}{\sigma^{2}_{r}} + n_{pr}\right)^{-1}, \qquad \mu_{*} = v_{*}\sum_{i \in \mathcal{I}_{pr}}(Z_{pir} + b_{ir}).$$

$$\boxed{\theta_{pr} \mid Z_{r}, b_{r}, \sigma^{2}_{r} \sim \mathcal{N}(\mu_{*}, v_{*}).}$$

**Where this lives in the code.** There is no per-node θ sampler. Every Gibbs $\theta$
draw (1PL, 2PL/3PL, GRM, independent or correlated) goes through one cross-node path:
`run_chain` builds each node's per-person statistics $(S_{pr}, n_{pr})$ from its latent
view (§1.5) with `missing_unit_stats`, `reduce_suff_stats_to_dims` collapses them onto
latent dimensions (adding the statistics of nodes that share a trait),
`build_sigma` assembles $\Sigma = DRD$, and `gibbs_theta_correlated` draws the vector
(`gibbs_theta_given` when some dimensions are updated by MH, §3.4). The scalar result
of this section is the **diagonal-$\Sigma$** special case of §2.6, which
`gibbs_theta_correlated` detects and computes elementwise. Read §2.6 as the general
statement; this section is the $R=1$ / diagonal reduction. (`compute_suff_stats`, the
older statistics function, is kept for compatibility; it cannot mask 3PL guess cells
or the binary items of a mixed node, so `run_chain` does not use it.)

**Why** $+b_{ir}$ appears. The mean of $Z$ is $\theta - b$, so $Z$ alone is not an observation of $\theta$. Adding $b$ back converts each $Z_{pir}$ into an observation of $\theta_{pr}$ with unit noise. The item parameter enters $\theta$'s update only through this adjustment.

------------------------------------------------------------------------

### 2.2 $b_{ir}$ — binary probit Rasch (1PL) node

Same model. Now $\theta_{r}$ is known and we update $b_{ir}$.

$$b_{ir} \sim \mathcal{N}(\mu_b, \tau^{2}).$$

We want $p(b_{ir} \mid Z_{r}, \theta_{r}, \tau^{2})$.

**Step 1.** Relevant persons: $\mathcal{P}_{ir} = \{p : T_r[p,i] = 1\}$. Let $n_{ir} = |\mathcal{P}_{ir}|$.

**Step 2.** From $Z_{pir} \sim \mathcal{N}(\theta_{pr} - b_{ir}, 1)$, define $w_{p} = \theta_{pr} - Z_{pir}$. Then $w_{p} = b_{ir} + \epsilon$, so $w_{p} \mid b_{ir} \sim \mathcal{N}(b_{ir}, 1)$. Template with $\mu_{0}=\mu_b$, $v_{0}=\tau^{2}$, $v=1$, $n=n_{ir}$.

**Step 3.**

$$v_{*} = \left(\frac{1}{\tau^{2}} + n_{ir}\right)^{-1}, \qquad \mu_{*} = v_{*}\left(\frac{\mu_b}{\tau^2} + \sum_{p \in \mathcal{P}_{ir}}(\theta_{pr} - Z_{pir})\right).$$

$$\boxed{b_{ir} \mid Z_{r}, \theta_{r}, \tau^{2} \sim \mathcal{N}(\mu_{*}, v_{*}).}$$

Implemented in `gibbs_samplers.gibbs_b()` (one node) and as the pooled draw over
every member of a shared b-group in `run_chain` (statistics summed over the members'
latent masks, prior counted once).

------------------------------------------------------------------------

### 2.3 $\theta_{pr}$ — binary probit 2PL or 3PL node

The 2PL augmented model is:

$$Z_{pir} \mid \theta_{pr}, b_{ir}, a_{ir} \sim \mathcal{N}(a_{ir}(\theta_{pr} - b_{ir}),\; 1).$$

For 3PL, once the Béguin-Glas indicator $U_{pir}$ is drawn, the relevant $Z$ values are exactly those with $U_{pir}=0$. Their distribution is this same 2PL form restricted to those cells. So the derivation is identical once we restrict to $U_{pir}=0$ pairs.

We want $p(\theta_{pr} \mid Z, b, a, \sigma^{2}_{r})$.

**Step 1.** $\theta_{pr}$ still appears linearly in the mean of $Z$. The likelihood is Gaussian in $\theta_{pr}$.

**Step 2.** Define $w'_{i} = Z_{pir} + a_{ir}b_{ir}$. Then $w'_{i} = a_{ir}\theta_{pr} + \epsilon$, so $w'_{i} \mid \theta_{pr} \sim \mathcal{N}(a_{ir}\theta_{pr}, 1)$. Each observation has a different "slope" $a_{ir}$. The template generalizes: each observation contributes precision $a_{ir}^{2}$ to the posterior precision.

**Step 3.**

$$v_{*} = \left(\frac{1}{\sigma^{2}_{r}} + \sum_{i \in \mathcal{I}_{pr}} a_{ir}^{2}\right)^{-1}, \qquad \mu_{*} = v_{*}\sum_{i \in \mathcal{I}_{pr}} a_{ir}(Z_{pir} + a_{ir}b_{ir}).$$

$$\boxed{\theta_{pr} \mid Z, b, a, \sigma^{2}_{r} \sim \mathcal{N}(\mu_{*}, v_{*}).}$$

The 1PL result is recovered by setting $a_{ir} = 1$ everywhere.

------------------------------------------------------------------------

### 2.4 $b_{ir}$ — binary probit 2PL or 3PL node

Same model. $\theta_{r}$ and $a_{r}$ are known. Prior: $b_{ir} \sim \mathcal{N}(\mu_b, \tau^{2})$.

Define $w'_{p} = a_{ir}\theta_{pr} - Z_{pir}$. Then $w'_{p} = a_{ir}b_{ir} + \epsilon$, mean $a_{ir}b_{ir}$, variance 1. Since $a_{ir}$ is item-level (constant across $p$):

$$v_{*} = \left(\frac{1}{\tau^{2}} + n_{ir}a_{ir}^{2}\right)^{-1}, \qquad \mu_{*} = v_{*}\left(\frac{\mu_b}{\tau^2} + a_{ir}\sum_{p \in \mathcal{P}_{ir}}(a_{ir}\theta_{pr} - Z_{pir})\right).$$

$$\boxed{b_{ir} \mid Z, \theta, a, \tau^{2} \sim \mathcal{N}(\mu_{*}, v_{*}).}$$

------------------------------------------------------------------------

### 2.5 $\theta_{pr}$ — GRM (polytomous probit) node

The GRM augmentation (Johnson-Albert 1999) is:

$$Z_{pir} \mid \theta_{pr} \sim \mathcal{N}(\theta_{pr},\; 1).$$

The observed category tells us which interval $Z$ fell in: $Y^{*}_{pir} = m$ iff $\tau_{i,m} < Z_{pir} \leq \tau_{i,m+1}$.

We want $p(\theta_{pr} \mid Z_{r}, \sigma^{2}_{r})$.

**The key difference from the binary case.** The mean of $Z$ is just $\theta_{pr}$. No item term. Item information lives in the thresholds, not the mean. So $Z_{pir}$ is directly an observation of $\theta_{pr}$ with unit noise. No adjustment needed.

**Apply the template** with $w_{i} = Z_{pir}$, $\mu_{0}=0$, $v_{0}=\sigma^{2}_{r}$, $v=1$, $n=n_{pr}$:

$$v_{*} = \left(\frac{1}{\sigma^{2}_{r}} + n_{pr}\right)^{-1}, \qquad \mu_{*} = v_{*}\sum_{i \in \mathcal{I}_{pr}} Z_{pir}.$$

$$\boxed{\theta_{pr} \mid Z_{r}, \sigma^{2}_{r} \sim \mathcal{N}(\mu_{*}, v_{*}).}$$

No $b$ term in $\mu_{*}$. Compare to the binary case where $\mu_{*}$ contains $\sum(Z_{pir} + b_{ir})$.

**Where this lives in the code.** GRM $\theta$ uses the same cross-node path as every
other node. The GRM-specific "no $b$ in the statistic" behaviour comes from the latent
view (§1.5), whose core at a GRM node is $\theta$ alone, so $S_{pr} = \sum_i Z_{pir}$
(less the covariate offset). At a mixed node the binary items contribute $Z + b$ and the
polytomous items $Z$. The draw itself is `gibbs_theta_correlated`.

------------------------------------------------------------------------

### 2.6 $\theta_{p}$ — correlated trait vector across $R$ nodes

The prior is now a joint MVN:

$$\theta_{p} = (\theta_{p1}, \ldots, \theta_{pR})^{\top} \sim \mathcal{N}_{R}(\mathbf{0}, \Sigma).$$

At each node $r$, the $Z$ values are Gaussian in $\theta_{pr}$ exactly as before. The difference is that the prior links $\theta_{pr}$ to $\theta_{pr'}$ across nodes through $\Sigma$. (Written here with one dimension per node. When nodes share a trait, the vector runs over the $D$ dimensions and the statistics $S_{pr}$, $n_{pr}$ of the nodes on a dimension add, `reduce_suff_stats_to_dims`.)

We want $p(\theta_{p} \mid Z, b, \Sigma)$.

**Step 1.** Write the log posterior. The prior contributes $-\frac{1}{2}\theta_{p}^{\top}\Sigma^{-1}\theta_{p}$. Each node $r$ contributes a sum of squared residuals in $\theta_{pr}$, exactly as in §2.1. Collecting:

$$\log p(\theta_{p} \mid \cdots) = -\frac{1}{2}\theta_{p}^{\top}(\Sigma^{-1} + \Lambda_{p})\theta_{p} + \theta_{p}^{\top}S_{p} + \text{const}$$

where $\Lambda_{p} = \mathrm{diag}(n_{p1}, \ldots, n_{pR})$ and $S_{p} = (S_{p1}, \ldots, S_{pR})^{\top}$ with $S_{pr} = \sum_{i \in \mathcal{I}_{pr}}(Z_{pir} + b_{ir})$.

**Step 2.** Any log-density quadratic in a vector is MVN. The quadratic $-\frac{1}{2}\theta^{\top}A\theta + b^{\top}\theta$ is the kernel of $\mathcal{N}_{R}(A^{-1}b,\; A^{-1})$, where $A$ is the **posterior precision matrix** and $A^{-1}$ is the posterior covariance. Here:

$$A = \Sigma^{-1} + \Lambda_{p} \qquad \text{(posterior precision = prior precision + likelihood precision)},$$

so the posterior covariance is the inverse of that sum — not the sum itself:

$$\Sigma_{*} = A^{-1} = (\Sigma^{-1} + \Lambda_{p})^{-1} \qquad \text{(posterior covariance = inverse of posterior precision)}.$$

**Step 3.** The posterior mean. The general form is $\mu_{p}^{*} = \Sigma_{*}(\Sigma^{-1}\mu_{0} + S_{p})$. Here $\mu_{0} = \mathbf{0}$, so:

$$\mu_{p}^{*} = \Sigma_{*}(\Sigma^{-1} \cdot \mathbf{0} + S_{p}) = \Sigma_{*} S_{p}.$$

Writing $\mu_{0}$ explicitly before substituting is important because in non-zero-mean prior specifications — for example, when person-level covariates shift the prior mean — $\Sigma^{-1}\mu_{0}$ would not vanish. The zero-mean simplification is a feature of the default specification, not of the general formula.

$$\boxed{\theta_{p} \mid Z, b, \Sigma \sim \mathcal{N}_{R}(\Sigma_{*}S_{p},\; \Sigma_{*}) \quad\text{with}\quad \Sigma_{*} = (\Sigma^{-1} + \Lambda_{p})^{-1}.}$$

**The two variances: what is fixed and what is free.**

The likelihood variance of $Z$ is 1. This is not a simplifying assumption — it is exact by definition of the probit model. $\varepsilon \sim \mathcal{N}(0, 1)$ is what makes $\Pr(Y^* = 1) = \Phi(\theta - b)$. A different variance would define a different (non-probit) link function. Consequently, $\Lambda_p = \mathrm{diag}(n_{p1}, \ldots, n_{pR})$ contains raw item counts, not counts divided by some free noise variance.

The prior variance $\sigma^2_d$ (the diagonal elements of $\Sigma$) is estimated by `gibbs_sigma2` on dimensions with a slope-free node and fixed at 1 on dimensions whose nodes all have free slopes (§1.4). The code constructs $\Sigma = DRD$ from the current $\sigma^2_d$ and the correlation matrix $R$, and passes that $\Sigma$ to this function, so $\Sigma_*$ uses the current values.

**Correlation blocks.** With `Constraints(theta_corr=[[...], ...])` the correlation matrix is block-diagonal: estimated within each declared block, zero across blocks, and dimensions in no block independent. Without it, all $D$ dimensions form one block. The θ draw above is unchanged (it takes whatever $\Sigma$ is current); only the correlation update (§3.3, §2.14) runs per block.

**Dimensions updated by MH.** When some dimensions carry a logit/GPCM node (§3.4), their θ is drawn by MH and the remaining dimensions are drawn from the Gaussian conditional given them (`gibbs_theta_given`): the same formula with the MH dimensions moved into the prior mean through $\Sigma^{-1}$.

**What** $\sigma^2_r$ actually measures — and its relationship to $b_{ir}$.

Since $\mathrm{Var}(\varepsilon) = 1$ is fixed, $\sigma^2_r = \mathrm{Var}(\theta_{pr})$ measures the population spread in units of the probit scale. This has a direct consequence for how item difficulties are interpreted.

An item with $b_{ir} = 1.5$ is hard relative to a population with $\sigma^2_r = 1$: most persons have $\theta_{pr}$ near zero, well below 1.5, so the item is answered correctly by only about $\Phi(0 - 1.5) = 7\%$ of a typical person. The same $b_{ir} = 1.5$ is only moderately difficult for a population with $\sigma^2_r = 4$: the population extends well beyond 1.5 and the correct-response rate is much higher. The $b$ parameters live on the same probit scale as $\theta$, and their substantive meaning depends on the population distribution.

This coupling means $\sigma^2_r$ and $b_{ir}$ are **not independent in the joint posterior**, even though their priors are independent. The same response pattern can be explained by expanding $\sigma^2_r$ (the population spreads out, reaching hard items) or contracting $b_{ir}$ (the items become easier). The Gibbs sampler handles this by conditioning: at each step $b$ is drawn given the current $\theta$ values, and $\theta$ (and hence $\sigma^2_r$) is drawn given the current $b$ values. The chain converges because the priors on both provide regularization, and because each draw is exact. Conditioning alone moves slowly along the common location of $\theta$ and $b$, though; the location move of §5.2 moves along it directly.

Concretely: if $\sigma^2_r$ takes a large draw in one iteration, $\theta$ values spread out, making $\sum_p(\theta_{pr} - Z_{pir})$ larger. The $b_{ir}$ posterior mean $\mu_* = v_* \sum_p(\theta_{pr} - Z_{pir})$ responds by shifting $b_{ir}$ upward (harder). In the next iteration, $\theta$ is drawn given the new, harder $b_{ir}$, pulling $\theta$ values back up. This back-and-forth is not a bug — it is the sampler correctly exploring the joint posterior of $(\theta, b, \sigma^2)$.

In practice this means: **individual** $b_{ir}$ estimates are interpretable only in conjunction with $\sigma^2_r$. When comparing item difficulties across groups or studies, researchers must either hold $\sigma^2_r$ fixed (fixing the scale) or normalise $b_{ir}$ by $\sigma_r$ before comparing. This is exactly the identification issue that arises whenever a latent variable model has a free variance component — a point that Albert (1992), Fox (2010), and the IRT measurement invariance literature address explicitly.

**Connection to the scalar case.** When $\Sigma$ is diagonal, $\Sigma^{-1} + \Lambda_{p}$ is diagonal too. The MVN draw factors into $R$ independent scalar draws, each identical to §2.1. The correlation enters only through the off-diagonal elements of $\Sigma^{-1}$. When $\rho \neq 0$, information about $\theta_{p1}$ from node 1's $Z$ values pulls the posterior of $\theta_{p2}$ in the correlated direction. Implemented in `gibbs_samplers.gibbs_theta_correlated()`.

------------------------------------------------------------------------

### 2.7 $\sigma^{2}_{r}$ — per-node trait variance, InvGamma prior

The "data" for $\sigma^{2}_{r}$ are the $P$ drawn values $\theta_{1r}, \ldots, \theta_{Pr}$. The $Z$ values do not appear here. Once $\theta$ is drawn, $Z$ adds no information about $\sigma^{2}_{r}$ because $\sigma^{2}_{r}$ does not enter the likelihood of $Z$ directly.

Apply the InvGamma template from §1.3 with prior $\mathrm{InvGamma}(\alpha_{0}, \beta_{0})$:

$$\boxed{\sigma^{2}_{r} \mid \theta_{\cdot r} \sim \mathrm{InvGamma}\!\left(\alpha_{0} + \frac{P}{2},\; \beta_{0} + \frac{1}{2}\sum_{p=1}^{P}\theta_{pr}^{2}\right).}$$

Implemented in `gibbs_samplers.gibbs_sigma2()`. The default hyperprior is $\mathrm{InvGamma}(1, 1)$. This draw runs only on dimensions whose variance is estimated (Rasch-type dimensions by default, §1.4); a fixed variance is used as given, and a HalfNormal prior on $\sigma_r$ switches to MH (§4).

------------------------------------------------------------------------

### 2.8 $\tau^{2}$ — item-difficulty variance, InvGamma prior

Same structure. The "data" are the $IR$ item difficulties $b_{ir}$. Prior $\mathrm{InvGamma}(\alpha_{0}^{\tau}, \beta_{0}^{\tau})$.

The package implements one $\tau^{2}$ per **b-sharing group** $g$ (a set of nodes whose item difficulties are constrained equal by the `Constraints` object; in the unshared case each node is its own group of size 1). Let $I_{g}$ denote the number of items in group $g$. The full conditional is:

$$\boxed{\tau^{2}_{g} \mid b_{g} \sim \mathrm{InvGamma}\!\left(\alpha_{0}^{\tau} + \frac{I_{g}}{2},\; \beta_{0}^{\tau} + \frac{1}{2}\sum_{i \in g}(b_{i} - \mu_b)^{2}\right).}$$

This is the same InvGamma template from §1.3 with the $I_{g}$ item difficulties in group $g$ playing the role of the $P$ person traits. Implemented in `gibbs_samplers.gibbs_b_variance()`.

**The pooled-τ² formula** (one value across all $I \times R$ difficulties) is the special case where all nodes are in one group:

$$\tau^{2} \mid b \sim \mathrm{InvGamma}\!\left(\alpha_{0}^{\tau} + \frac{IR}{2},\; \beta_{0}^{\tau} + \frac{1}{2}\sum_{r=1}^{R}\sum_{i=1}^{I}b_{ir}^{2}\right).$$

The per-group formula reduces to this when $g$ contains all items across all nodes.

------------------------------------------------------------------------

### 2.9 $\Sigma$ — independent case as block-diagonal

When traits are independent (`FixedCorrelation(rho=0)`; the default is `LKJPrior(eta=1)`), $\Sigma = \mathrm{diag}(\sigma^{2}_{1}, \ldots, \sigma^{2}_{D})$. There is no joint update. Each estimated $\sigma^{2}_{d}$ is updated independently via §2.7, and `gibbs_theta_correlated` takes its elementwise path.

With an estimated correlation there are two priors. The default separation strategy $\Sigma = DRD$ keeps the variances (InvGamma, §2.7) and the correlation matrix (LKJ, MH, §3.3) factored. An `InverseWishartPrior` on `theta.correlation` instead draws each block's full covariance in one conjugate step (§2.14); its prior on variances and correlations is entangled, so it is opt-in.

### 2.10 $\tau_{i,k}$ — GRM thresholds, Normal prior

**Setup.** At a GRM node, $Z_{pir} \sim \mathcal{N}(\theta_{pr} + \text{offset}_{pir}, 1)$ and $Y^{*}_{pir} = m$ iff $\tau_{i,m-1} < Z_{pir} \leq \tau_{i,m}$, with $\tau_{i,-1} = -\infty$, $\tau_{i,M-1} = +\infty$. The free thresholds must satisfy $\tau_{i,0} < \tau_{i,1} < \cdots < \tau_{i,M-2}$ (0-based, as in the code).

**Prior.** $\tau_{i,k} \sim \mathcal{N}(\mu_0, \tau^2)$ independently, restricted to increasing order. $\tau^2$ is the current b-group variance when it is estimated.

We want $p(\tau_{i,k} \mid Z, Y^{*}, \tau_{i,\neg k})$.

**Step 1. Where does $\tau_{i,k}$ appear?** Write the joint density of $(Z_{\cdot i}, Y^{*}_{\cdot i})$ given $\theta$ and the thresholds:

$$\prod_{p} \mathcal{N}(Z_{pir};\, \theta_{pr} + \text{offset}_{pir},\, 1)\;\mathbb{1}\{\tau_{i,Y^{*}_{pir}-1} < Z_{pir} \le \tau_{i,Y^{*}_{pir}}\}.$$

The Normal density of $Z$ does not involve $\tau$ at all. The thresholds enter **only** through the indicators. So given $Z$ and $Y^{*}$, the likelihood of $\tau_{i,k}$ is flat on the set where every indicator equals 1, and zero outside it. (The latent responses are not noisy measurements of $\tau_{i,k}$; no Normal likelihood term built from them belongs in this conditional.)

**Step 2. The allowed interval.** Threshold $k$ separates category $k$ (below) from category $k+1$ (above). Every indicator is satisfied iff

$$L_k = \max\!\bigl(\tau_{i,k-1},\; \max\{Z_{pir} : Y^{*}_{pir} = k\}\bigr) \;<\; \tau_{i,k} \;\le\; U_k = \min\!\bigl(\tau_{i,k+1},\; \min\{Z_{pir} : Y^{*}_{pir} = k+1\}\bigr).$$

An empty category contributes no bound, so an item or category with no data gets its prior truncated by its neighbours.

**Step 3. Multiply by the prior.**

$$\boxed{\tau_{i,k} \mid Z, Y^{*}, \tau_{i,\neg k} \sim \mathcal{N}(\mu_{0}, \tau^{2}) \text{ truncated to } (L_{k}, U_{k}).}$$

Thresholds are drawn in order $k = 0, \ldots, M-2$ within each item (Gibbs within the item), vectorized over items, with the tail-stable log-space inverse CDF of `truncnorm_draw`. Under a flat prior ($\tau^2 \to \infty$) this is $\mathrm{Uniform}(L_k, U_k)$, the Albert and Chib (1993) result.

**Why ordering is preserved automatically.** $L_{k} < U_{k}$ holds whenever the current $Z$ was drawn consistently with the current thresholds, which `sample_Z_grm()` guarantees.

**Why this alone mixes badly, and what runs before it.** With many respondents the interval $(L_k, U_k)$ is very narrow, so each exact draw barely moves $\tau_{i,k}$ (Cowles, 1996). Each iteration therefore first runs a Cowles step on all thresholds of the item with $Z$ integrated out (§5.1), redraws $Z$, and only then makes the exact draw above. Shared thresholds are pooled over every member node, both in the Cowles step and in the bounds.

Implemented in `gibbs_samplers.gibbs_thresholds_grm()`. Prior parameters are resolved from `priors.resolve("b.poly", r)` at the call site.

## 2.11 Covariate regression coefficients ($\beta$, $\kappa$, $\gamma$) and missing covariates

The augmented model at a covariate-enabled probit node is the latent view of §1.5 with

$$\text{offset}_{pir} = \underbrace{X_{p}^{\top}\beta}_{\text{person}} + \underbrace{Q_{i}^{\top}\kappa}_{\text{item}} + \underbrace{W_{pi}^{\top}\gamma}_{\text{person}\times\text{item}}, \qquad Z_{pir} = a_{ir}\bigl(\text{core}_{pir} + \text{offset}_{pir}\bigr) + \epsilon_{pir}.$$

All three covariate terms add to the ability side of the predictor, inside the discrimination. The sign of an estimated coefficient encodes the direction of the effect directly: a positive entry raises the linear predictor and so the probability of the node's "1" branch. No sign convention is baked in. Each node uses only the columns it selects (`person_covariates`, `item_covariates`, `pi_covariates`).

**Groups, not nodes.** Coefficients are keyed by group: one $\beta$ per latent dimension that carries person covariates (nodes sharing a trait share its $\beta$), one $\kappa$ per b-sharing group, one $\gamma$ per `gamma_shared` group (one per node by default). §2.11.4 explains why sharing forces pooling.

**Priors.** Each group's coefficient vector has an isotropic Normal prior with its own variance, estimated by default:

$$c_g \sim \mathcal{N}_{K}(\mu_{0}\mathbf{1}, \tau^{2}_{g}I_{K}), \qquad \tau^2_g \sim \mathrm{InvGamma}(1, 1) \;\text{(Gibbs, §4a)}, \qquad c \in \{\beta, \kappa, \gamma\}.$$

### 2.11.1 One Gaussian regression for all three coefficient types

**The residual.** To update coefficient group $c$ at node $r$, remove from $Z$ everything except its own term:

$$E_{r}[p,i] = Z_{pir} - a_{ir}\bigl(\text{core}_{pir} + \text{other covariate terms}\bigr) = a_{ir}\,d_{pi}^{\top}c + \epsilon_{pir},$$

on node $r$'s latent mask, with design row $d_{pi} = X_p$ (β), $Q_i$ (κ) or $W_{pi}$ (γ). This is the template of §1.2 with slope $a_{ir}$, as in §2.3.

**Step 1. Sufficient statistics.** Expanding $\sum (E - a\,d^\top c)^2$ in $c$ and summing over every node $r$ of the group and every cell of its mask:

$$G = \sum_{r}\sum_{(p,i)} a_{ir}^{2}\, d_{pi}d_{pi}^{\top} \in \mathbb{R}^{K\times K}, \qquad h = \sum_{r}\sum_{(p,i)} a_{ir}\, d_{pi}\,E_{r}[p,i] \in \mathbb{R}^{K},$$

so the log-likelihood is $h^\top c - \tfrac12 c^\top G c + \text{const}$. The design is reduced along the axis it does not vary on: per person for β (row sums $\sum_i a_{ir}^2$ and $\sum_i a_{ir}E$), per item for κ, per cell for γ.

**Step 2. Add the prior.**

$$\boxed{c \mid \text{rest} \sim \mathcal{N}_K\!\left(V_*\Bigl(\frac{\mu_0}{\tau^2_g}\mathbf{1} + h\Bigr),\; V_*\right), \qquad V_* = \Bigl(\frac{I_K}{\tau^2_g} + G\Bigr)^{-1}.}$$

**For** $K=1$ with one Rasch node and β this reduces to

$$v_{*} = \Bigl(\frac{1}{\tau^{2}} + \sum_{p} n_{p} X_{p}^{2}\Bigr)^{-1}, \qquad \mu_{*} = v_{*}\Bigl(\frac{\mu_{0}}{\tau^{2}} + \sum_{p} X_{p} S_{p}\Bigr),$$

with $n_p$ the person's relevant items and $S_p = \sum_i E_r[p,i]$: the §1.2 template with observations $X_p$, each with weight $n_p$.

Each node keeps its **own** discrimination vector, so a group that mixes Rasch and 2PL nodes, or 2PL nodes in different a-groups, is exact. Implemented in `gibbs_samplers.coef_gaussian_terms()` (the statistics) and `gibbs_samplers.gibbs_coef_pooled()` (the draw). A group with a logit/GPCM member is updated by MH with this Gaussian part plus the logit likelihood (§3.4).

**Missing covariate values.** Covariates with missing entries are completed by their current draws (§2.11.4) before these statistics are built, so every cell enters the likelihood (full imputation); no observed-cell mask is applied.

### 2.11.2 Missing covariate values as parameters ($X$, $Q$, $W$)

A missing covariate entry is a parameter, drawn from its full conditional each iteration. Missingness is per **feature**: only the missing features of a unit are drawn, conditional on its observed features. The units are persons ($X$), items ($Q$) or cells ($W$). The prior is independent Normal per feature, $\mathcal{N}(\mu_X, \sigma_X^2)$, by default $\mathcal{N}(0,1)$ (standardized covariates). The prior mean and variance are fixed numbers.

**Step 1. Per-node statistics of the unit.** At node $r$, remove everything except the term of this covariate type, $E_r = a_{r}\,(x_u^{\top} c_r) + \epsilon$ on the unit's cells, and aggregate over them:

$$n_{ru} = \sum_{\text{cells of } u} a_{ir}^{2}, \qquad S_{ru} = \sum_{\text{cells of } u} a_{ir}\,E_r[p,i]$$

(a person's items for $X$, an item's persons for $Q$, the single cell for $W$).

**Step 2. Split observed and missing features.** With $x_u = (x_o, x_m)$ and $c_r = (c_{r,o}, c_{r,m})$, the observed part is known, so $S_{ru} - n_{ru}\,x_o^\top c_{r,o}$ is a Gaussian observation of $x_m^\top c_{r,m}$ with precision $n_{ru}$.

**Step 3. Pool over nodes and add the prior.** The nodes are conditionally independent given $Z$, so precisions and linear statistics add:

$$\Lambda_u = \frac{I}{\sigma_X^2} + \sum_{r} n_{ru}\,c_{r,m}c_{r,m}^{\top}, \qquad \ell_u = \frac{\mu_X}{\sigma_X^2}\mathbf{1} + \sum_{r} \bigl(S_{ru} - n_{ru}\,x_o^{\top}c_{r,o}\bigr)\,c_{r,m}.$$

$$\boxed{x_{m} \mid \text{rest} \sim \mathcal{N}(\Lambda_u^{-1}\ell_u,\; \Lambda_u^{-1}).}$$

Units are batched by missingness pattern (units missing the same features share the shapes), so the draws are two LAPACK calls per pattern; the RNG stream differs from a per-unit loop, the target does not. If a unit has no relevant cell at any node using the covariate, the draw is from the prior. Implemented in `missing_unit_stats`, `missing_terms` and `draw_from_terms`, wrapped by `gibbs_X_missing`, `gibbs_Q_missing` and `gibbs_W_missing`. Units that a logit/GPCM node uses are updated by MH with this Gaussian part plus the logit likelihood (`_mh_missing_units`, §3.4).

### 2.11.3 The Z-residualization requirement

Every update in §2 is correct only if it receives a residual from which the other blocks' contributions have been removed. $\theta$ and $b$ share $Z$ with the coefficients, so this matters in both directions.

**The failure it prevents.** If raw $Z_{pir}$ were passed to the $\theta$ update (§2.1), its statistic would be

$$S_{pr} = \sum_{i}(Z_{pir} + b_{ir}) = n_{pr}\theta_{pr} + n_{pr}X_{p}^{\top}\beta + \text{noise}.$$

The $\theta$ draw would absorb $X_{p}^{\top}\beta$, the $\beta$ residual $Z - (\hat{\theta} - b)$ would then be pure noise, and $\beta$ would be estimated near zero however long the chain. The sampler would run without error.

**The scale it happens on.** $Z$ lives on the $a$-scale: its mean is $a_{ir}(\text{core} + \text{offset})$. A draw that isolates the core (and re-applies $a_{ir}$ itself) must subtract $a_{ir}\cdot\text{offset}$, not the raw offset:

$$E_{pir} = Z_{pir} - a_{ir}\,\text{offset}_{pir}.$$

For Rasch nodes $a_{ir}=1$ and this is the raw subtraction. The latent view (§1.5) does this for every block: `_latent_resid(r, include)` returns $Z_r - a_r(\text{core}_r + \text{included covariate terms})$, and each block adds back what it updates. Functions that apply $a_{ir}$ themselves (`sample_Z`, `log_prob`, `recompute_node`, `gibbs_a`) receive the **raw** offset. In particular `gibbs_a` regresses $Z$ on the full ability predictor $(\theta - b + \text{offset})$ (§3.1a); omitting the offset from that regressor makes $a$ absorb the covariate signal, the same failure as for $\theta$.

**Scan order within one iteration** (the `run_chain` docstring has the full list):

1. Node loop: $Z$ (and $U$ at 3PL nodes) → node-level $b$, $c$ and $a$ for unshared parameters; at GRM nodes the Cowles step (§5.1), then $Z$, then the exact threshold draw (§2.10).
2. Shared $a$ and $c$ (pooled), uncorrelated a-group variances (§4a).
3. Shared $b$ (pooled Gibbs, or item-wise MH with a logit member), pooled thresholds.
4. $\theta$ (Gibbs, or person-wise MH on dimensions with a logit/GPCM node).
5. Location move (§5.2).
6. $\sigma^2$, θ correlation per block, b-group variances.
7. b-correlation groups, then $\Sigma_b$ (§2.12).
8. a-correlation groups, their variances, then $\Sigma_a$ (§2.13).
9. Covariate coefficients (§2.11.1), interweaving (§5.3), missing covariates (§2.11.2).

### 2.11.4 Pooled covariate coefficients (shared across nodes)

When a covariate is declared on nodes that **share a parameter** (a shared latent
dimension for $\beta$, a shared b-group for $\kappa$, a `gamma_shared` group for
$\gamma$), the coefficient is a single vector estimated from all member nodes at once.
§2.11.1 is already written for this case: $G$ and $h$ are sums over the member nodes,
with the prior counted **once**. Because the nodes are conditionally independent given
$Z$, their Gaussian likelihoods multiply, so precisions and linear statistics add.

**Why this must be pooled, not drawn per node.** Drawing the shared coefficient once per
member node (each conditioning on only that node's cells, the last overwriting the rest)
is not a full conditional of the joint target; it silently discards every node's data
but the last and widens the posterior.

**Covariate-on-shared-parameter validation.** A covariate that explains a shared
parameter must be declared **identically** on every node sharing it (same feature
selection), because one coefficient cannot be built from two different design matrices.
`data_prep._group_and_validate` raises if the declarations differ or are only partially
present. Pooling γ is opt-in: γ is identified at every node, so `gamma_shared` is a
modelling assertion, not a requirement.

**Historical note (why the ordering matters).** Two convention bugs were latent here
until 2PL-with-covariates was first run: `ProbitTwoPLNode.log_prob` added the offset
*after* the $a$-scaling while `sample_Z` added it *before* (fixed: both now add before),
and `gibbs_a` regressed on $(\theta-b)$ with no offset at all (fixed: offset now in the
regressor). Both would silently drive covariate coefficients toward zero.

### 2.11.5 Identification of the coefficients

Exact-Gibbs status is a statement about the *draw*, not about *identification*.

- **β** (person covariates) is identified with θ, whose prior mean is fixed at 0 (§1.4).
  Centring $X$ makes β and θ nearly orthogonal and improves mixing.
- **κ** (item covariates) shares the item axis with a free $b$: the likelihood depends on
  $(b, \kappa)$ only through $b_i - Q_i^\top\kappa$. κ is still identified, through
  $b$'s prior: with $b_i \sim \mathcal{N}(\mu_b, \tau^2)$, integrating $b$ out gives
  $b_i - Q_i^\top\kappa \sim \mathcal{N}(\mu_b - Q_i^\top\kappa, \tau^2)$, a regression of the
  item locations on $Q$. What looked like a "collapse" of κ in earlier runs was the slow
  mixing of a two-block Gibbs sampler along the ridge $b - Q\kappa = \text{const}$ (Liu,
  Wong and Kong, 1994), not non-identifiability. The interweaving step of §5.3 removes it.
  (`docs/TESTING.md` §5 has the probes that separate code, identifiability and mixing.)
- **γ** (person-by-item covariates) varies over cells, so it is identified at every node.

## 2.12 $b_{i}$ — cross-node correlated item difficulties

When a b-correlation group of $G$ b-sharing groups is declared (`Constraints(b_corr=...)`), the prior on item $i$'s difficulty vector is:

$$b_{i} = (b_{i,g_1}, \ldots, b_{i,g_G})^{\top} \sim \mathcal{N}_{G}(\boldsymbol\mu_b,\; \Sigma_{b}),$$

with $\boldsymbol\mu_b$ the groups' prior means (0 by default). The draw below is written for $\boldsymbol\mu_b = \mathbf 0$; with non-zero means the code draws the centred $b_i - \boldsymbol\mu_b$ (statistic $S_i - \Lambda_i\boldsymbol\mu_b$) and adds $\boldsymbol\mu_b$ back.

Every node $r$ of group $g_k$ contributes, on its latent mask (§1.5), $Z_{pir} = a_{ir}(\theta_{pr} - b_{i,g_k} + \text{offset}_{pir}) + \epsilon$. The statistics of item $i$ for group $g_k$ are summed over **all** member nodes of the group (the $U=0$ cells at 3PL nodes, the $a$-scaled offset removed):

$$S_{i,g_k} = \sum_{r \in g_k}\sum_{p} a_{ir}\bigl(a_{ir}(\theta_{pr} + \text{offset}_{pir}) - Z_{pir}\bigr), \qquad n_{i,g_k} = \sum_{r \in g_k}\sum_{p} a_{ir}^2,$$

the sums running over the mask; for Rasch members $a_{ir} = 1$. Stack across groups:

$$\Lambda_i = \mathrm{diag}(n_{i,g_1}, \ldots, n_{i,g_G}), \qquad S_i = (S_{i,g_1}, \ldots, S_{i,g_G})^{\top}.$$

**Full conditional** (MVN-Normal conjugacy, identical to §2.6 with items in the role of persons):

$$A_i = \Sigma_{b}^{-1} + \Lambda_{i} \qquad \text{(posterior precision = prior precision + likelihood precision)},$$

$$\Sigma_{i}^{*} = A_i^{-1} = (\Sigma_{b}^{-1} + \Lambda_{i})^{-1}, \qquad \mu_{i}^{*} = \Sigma_{i}^{*}\,S_i.$$

$$\boxed{b_{i} \mid Z, \theta, \Sigma_{b} \sim \mathcal{N}_{G}(\mu_{i}^{*},\; \Sigma_{i}^{*}).}$$

$I$ independent $G$-dimensional MVN draws (one per item). When $G=1$ (no correlation), $\Lambda_i = [n_{i}]$ and the formula reduces exactly to the scalar §2.2 result.

**Skip-scalar guard.** In the Gibbs scan, the per-node scalar `gibbs_b` is skipped for nodes in a b-correlation group. Only the joint draw runs. Running both would constitute two $b$ updates per iteration, biasing the chain.

Implemented in `gibbs_samplers.gibbs_b_correlated_from_stats()` (statistics from `_b_gauss_stats` in `run_chain`), batched over items. When some groups have a logit/GPCM member, the probit groups are drawn jointly given the logit groups (`gibbs_theta_given`, the same algebra), and each logit group is updated by item-wise MH whose prior is the conditional $b_{ik} \mid b_{i,-k}$ from $\Sigma_b$ (§3.4). `gibbs_b_correlated_rng`, the older form that used only the first node of each group, is kept for compatibility.

------------------------------------------------------------------------

## 2.13 $a_{i}$ — cross-node correlated discrimination (TruncNormal prior)

When an a-correlation group of $G$ a-sharing groups is declared and the slopes have a truncated-Normal prior, item $i$'s discrimination vector has the prior

$$a_{i} = (a_{i,g_1}, \ldots, a_{i,g_G})^{\top} \sim \mathcal{N}_{G}(\mu_{a},\; \Sigma_{a}) \text{ truncated to } (0,\infty)^G.$$

Given $Z$ and the ability predictor $d_{pir} = \theta_{pr} - b_{ir} + \text{offset}_{pir}$, each member node of group $g_k$ contributes $Z_{pir} = a_{i,g_k} d_{pir} + \epsilon$ on its latent mask. Summed over the group's members:

$$s_{i,k} = \sum_{r \in g_k}\sum_{p} d_{pir}\,Z_{pir}, \qquad n_{i,k} = \sum_{r \in g_k}\sum_{p} d_{pir}^{2}.$$

**Componentwise exact Gibbs.** The joint conditional is a truncated MVN, which has no direct sampler. Its one-dimensional conditionals do. With $\Lambda = \Sigma_a^{-1}$, the prior of $a_{i,k}$ given the other components is Normal with precision $\Lambda_{kk}$ and linear term $\Lambda_{kk}\mu_k - \sum_{h \ne k}\Lambda_{kh}(a_{i,h} - \mu_h)$. Adding the likelihood terms:

$$\boxed{a_{i,k} \mid a_{i,-k}, Z, \theta, b \sim \mathcal{N}\!\left(\frac{\ell_{i,k}}{\Lambda_{kk} + n_{i,k}},\; \frac{1}{\Lambda_{kk} + n_{i,k}}\right) \text{ truncated to } (0, \infty), \qquad \ell_{i,k} = \Lambda_{kk}\mu_k - \sum_{h\ne k}\Lambda_{kh}(a_{i,h}-\mu_h) + s_{i,k}.}$$

Cycling $k = 1, \ldots, G$ (vectorized over items, `truncnorm_draw`) is a valid Gibbs scan on the truncated MVN. It needs no rejection step, so the correlation is respected on every iteration (the older joint rejection sampler, `gibbs_a_correlated`, is kept for compatibility). A group with a logit/GPCM member is updated by per-item MH on $\log a$ with the same conditional prior (§3.4).

**Group variances and the correlation.** Under an LKJ (or fixed) `a.correlation`, $\Sigma_a = D R D$ with $D$ the group SDs. When the slopes use the package-default prior, each correlated group's variance is estimated with a Half-Normal(0, 0.5) prior on its SD (MH on $\log v$ with the MVN likelihood of the centred slopes): held fixed at 0.5, the variances would push $\rho$ toward $\pm 1$ whenever the slopes vary less than that. The correlation update (§3.3) evaluates $\mathcal{N}(0, DRD)$ on the slopes **centred at the prior mean** $\mu_a$; raw slopes near 1 would read as strong co-movement.

**A stated approximation.** The truncated MVN prior has normalizing constant $\Pr_{\Sigma_a}(a_i > 0)$, which depends on $\Sigma_a$. The updates of the a-correlation, of the correlated group variances and of an Inverse-Wishart $\Sigma_a$ ignore it, so they target the posterior of the untruncated hyperparameters. The effect is small when $\mu_a$ is several prior SDs above 0 and grows as the prior puts mass near 0. (The uncorrelated a-group variance of §4a does include its normalizer.)

**Connection to §2.12.** The likelihood algebra is the same with $d_{pir}$ in the role of the slope; the difference is the positive orthant, which is why the draw is componentwise.

------------------------------------------------------------------------

## 2.14 $\Sigma$ — full covariance matrix, Inverse-Wishart prior

When an `InverseWishartPrior` is used for `theta.correlation`, `b.correlation`, or `a.correlation`, the full covariance matrix $\Sigma$ is drawn from its conjugate full conditional in one step.

**Prior.** $\Sigma \sim \mathrm{InvWishart}(\Psi_0,\, \nu_0)$, with density $p(\Sigma) \propto |\Sigma|^{-(\nu_0+G+1)/2}\exp(-\tfrac{1}{2}\mathrm{tr}(\Psi_0\Sigma^{-1}))$.

**Data.** Let $\{x_1, \ldots, x_N\}$ be $N$ independent observations from $\mathcal{N}_G(\mu_0, \Sigma)$. In the package:

- **theta.correlation**: $x_j = \theta_p$ restricted to the dimensions of one correlation block, $N = P$, $\mu_0 = \mathbf{0}$; one draw per block, and the block's $\sigma^2_d$ are read from the diagonal.
- **b.correlation**: $x_j = b_i$, $N = I$, $\mu_0 = \mathbf{0}$.
- **a.correlation** (TruncNormal): $x_j = a_i$, $N = I$, $\mu_0 = \mu_a$ (prior mean); this ignores the orthant normalizer (§2.13).

**Scatter matrix.** $S = \sum_{j=1}^{N}(x_j - \mu_0)(x_j - \mu_0)^{\top}$.

**Derivation.** Multiply prior by MVN likelihood and collect $\Sigma$ terms:

$$p(\Sigma \mid x_{1:N}) \propto |\Sigma|^{-(\nu_0+N+G+1)/2}\exp\!\left(-\tfrac{1}{2}\mathrm{tr}((\Psi_0+S)\Sigma^{-1})\right).$$

That is the kernel of an InvWishart.

$$\boxed{\Sigma \mid x_{1:N} \sim \mathrm{InvWishart}(\Psi_0 + S,\; \nu_0 + N).}$$

One `scipy.stats.invwishart.rvs()` call per iteration — no MH, no tuning.

**Trade-off vs the DRD separation.** When $\Sigma = DRD$ (diagonal $D$, correlation matrix $R$), the InvGamma Gibbs steps for each variance $\sigma^2_d$ and the MH step for $R$ are separate. The InvWishart replaces both with a single draw. The cost is that the prior on variances and the prior on correlations are entangled: shrinking correlations toward zero also affects the variance estimates. For this reason the DRD approach is the default and InvWishart is opt-in.

Implemented in `gibbs_samplers.gibbs_Sigma_invwishart()`.

------------------------------------------------------------------------

## 3. Non-conjugate parameters (Metropolis-Hastings)

For these parameters no closed-form posterior exists. We derive the log posterior and use it as the MH target.

### 3.1 $a_{ir}$ — 2PL/3PL discrimination, LogNormal prior (opt-in, MH)

The default slope prior is the truncated Normal of §3.1a, which is conjugate. This section is the LogNormal alternative, kept because IRT practice often states slope priors on the log scale.

**Model.** Augmented likelihood: $Z_{pir} \mid \theta, b, a \sim \mathcal{N}(a_{ir}d_{pir}, 1)$ with $d_{pir} = \theta_{pr} - b_{ir} + \text{offset}_{pir}$.\
Prior: $a_{ir} \sim \mathrm{LogNormal}(\mu_{a}, \sigma^{2}_{a})$, i.e., $\log a_{ir} \sim \mathcal{N}(\mu_{a}, \sigma^{2}_{a})$.

We want $p(a_{ir} \mid Z, \theta, b)$. The goal is to show why conjugacy fails even though $Z$ is Gaussian.

**Step 1.** Write the likelihood as a function of $a_{ir}$:

$$\log p(Z \mid a_{ir}) = -\frac{1}{2}\sum_{p}(Z_{pir} - a_{ir}d_{pir})^{2} + \text{const} = -\frac{1}{2}\left[a_{ir}^{2}\sum_{p}d_{pir}^{2} - 2a_{ir}\sum_{p}Z_{pir}d_{pir}\right] + \text{const}.$$

This is quadratic in $a_{ir}$: a Gaussian-looking kernel on the linear scale of $a$.

**Step 2.** Write the LogNormal prior as a density in $a$:

$$\log p(a_{ir}) = -\log a_{ir} - \frac{(\log a_{ir} - \mu_{a})^{2}}{2\sigma^{2}_{a}} + \text{const}.$$

This is quadratic in $\log a_{ir}$: a Gaussian kernel on the log scale of $a$.

**Step 3.** Combine. The log full conditional contains a quadratic in $a_{ir}$ (from the likelihood) and a quadratic in $\log a_{ir}$ (from the prior). These two scales do not add to a recognizable kernel, so no closed form exists.

**Step 4. The MH target on the working scale.** The chain moves in $\xi = \log a_{ir}$ with a symmetric random walk, so the target must be the density of $\xi$: the $a$-scale posterior times $|da/d\xi| = a$. That Jacobian cancels the $-\log a$ of the LogNormal density, leaving a Normal kernel in $\xi$:

$$\boxed{\log \pi(\xi_{ir}) = -\frac{1}{2}e^{2\xi_{ir}}\sum_{p}d_{pir}^{2} + e^{\xi_{ir}}\sum_{p}Z_{pir}d_{pir} - \frac{(\xi_{ir} - \mu_{a})^{2}}{2\sigma^{2}_{a}} + \text{const}.}$$

Writing the target with the $-\log a$ term *and* a symmetric proposal on $\log a$ (without the $+\log a$ Jacobian) would target the posterior times $1/a$ and bias the slopes down.

All $I$ items are proposed together and accepted item by item (they are conditionally independent given $Z$, θ and $b$). With `common_discrimination=True` one $\xi$ is proposed for the whole node and accepted on the sum over every cell, with the prior counted once. Implemented in `mcmc_sampler._mh_discrimination()`, used when the resolved prior is LogNormal. (An unbounded prior on $a$ is rejected outright: it leaves the reflection $(\theta, a, b) \to (-\theta, -a, -b)$ unbroken, so the model would not be identified.)

### 3.1a $a_{ir}$ — 2PL/3PL discrimination, TruncNormal prior (default, Gibbs)

**Prior.** $a_{ir} \sim \mathcal{N}(\mu_{a}, \sigma^{2}_{a})$ truncated to $(0, \infty)$.

This changes the prior from a Gaussian kernel in $\log a_{ir}$ (LogNormal) to a Gaussian kernel in $a_{ir}$ (TruncNormal). The likelihood is already a Gaussian kernel in $a_{ir}$ (Step 1 above). Now both terms are quadratic in $a_{ir}$ — conjugacy holds.

**Step 1.** Same likelihood as §3.1:

$$\log p(Z \mid a_{ir}) = -\frac{1}{2}\left[a_{ir}^{2}\sum_{p}d_{pir}^{2} - 2a_{ir}\sum_{p}Z_{pir}d_{pir}\right] + \text{const.}$$

**Step 2.** TruncNormal prior on $a_{ir} > 0$:

$$\log p(a_{ir}) = -\frac{(a_{ir} - \mu_{a})^{2}}{2\sigma^{2}_{a}} + \text{const} = -\frac{a_{ir}^{2}}{2\sigma^{2}_{a}} + \frac{\mu_{a}\,a_{ir}}{\sigma^{2}_{a}} + \text{const.}$$

(The truncation normalising constant does not depend on $a_{ir}$ and absorbs into the constant.)

**Step 3.** Apply the Normal–Normal template (§1.2). The sum of the two quadratic kernels is:

$$\log p(a_{ir} \mid Z, \theta, b) = -\frac{1}{2}\left(\frac{1}{\sigma^{2}_{a}} + \sum_{p}d_{pir}^{2}\right)a_{ir}^{2} + \left(\frac{\mu_{a}}{\sigma^{2}_{a}} + \sum_{p}Z_{pir}d_{pir}\right)a_{ir} + \text{const.}$$

Read off the posterior parameters:

$$v_{*} = \left(\frac{1}{\sigma^{2}_{a}} + \sum_{p \in \mathcal{P}_{ir}}d_{pir}^{2}\right)^{-1}, \qquad \mu_{*} = v_{*}\left(\frac{\mu_{a}}{\sigma^{2}_{a}} + \sum_{p \in \mathcal{P}_{ir}}Z_{pir}d_{pir}\right).$$

This is a Gaussian kernel in $a_{ir}$. Combined with the $a_{ir} > 0$ constraint from the prior:

$$\boxed{a_{ir} \mid Z, \theta, b \sim \mathcal{N}(\mu_{*}, v_{*}) \text{ truncated to } (0, \infty).}$$

**Sampling via inverse-CDF.** For scalar $a_{ir}$, draw from the truncated Normal directly:

$$U \sim \mathrm{Uniform}\!\left(\Phi\!\left(\frac{0 - \mu_{*}}{\sqrt{v_{*}}}\right),\; 1\right), \qquad a_{ir} = \mu_{*} + \sqrt{v_{*}}\,\Phi^{-1}(U).$$

No accept-reject needed. Vectorised across all $I$ items in a single numpy call. The procedure is identical for 2PL nodes (full $T_r$ index sets) and 3PL nodes (restricted to ability cells $T_{ab}$ where $U_{pir}=0$, exactly as the LogNormal MH branch uses).

**With covariates.** When covariates are active the ability predictor is
$d_{pir} = \theta_{pr} - b_{ir} + \text{offset}_{pir}$ (raw offset, since `gibbs_a`
applies $a$ itself). The `offset` argument to `gibbs_a` supplies this; the
statistics $\sum_p d_{pir}^2$ and $\sum_p Z_{pir}d_{pir}$ use the offset-inclusive
$d_{pir}$. Omitting the offset makes $a$ absorb the covariate signal (§2.11.5).

**Default prior.** $\mathcal{N}(1, 0.5)$ truncated to $(0, \infty)$.

**One slope for the whole node** (`common_discrimination=True`). The node has a single $a_r$ shared by all its items, so it appears in every cell, and its full conditional pools every cell of the node: the same truncated Normal with the sums over all items,

$$v_* = \Bigl(\frac{1}{\sigma_a^2} + \sum_p\sum_i d_{pir}^2\Bigr)^{-1}, \qquad \mu_* = v_*\Bigl(\frac{\mu_a}{\sigma_a^2} + \sum_p\sum_i d_{pir}Z_{pir}\Bigr).$$

The single draw is returned repeated over the items. A node-wide slope inside an `a_shared` or `a_corr` group, or with an estimated slope variance, is rejected with an error (one slope per node cannot also be one slope per item of a group).

**Shared $a$.** For an $a$-sharing group (as opposed to an $a$-correlation group,
§2.13), the same pooling rule as $b$/$\kappa$ applies: sufficient statistics
$\sum_p d^2$ and $\sum_p dZ$ are summed across the group's member nodes and the
truncated-Normal (Gibbs) or LogNormal (MH) draw is done once for the shared vector.
When a member is logit, the shared slopes are updated by item-wise MH whose target adds
the logit members' likelihood to the probit members' Gaussian term (§3.5).

Implemented in `gibbs_samplers.gibbs_a()` (`common=True` for the node-wide slope), on the node's latent mask (every relevant cell at a 2PL node, the $U=0$ cells at a 3PL node).

------------------------------------------------------------------------

### 3.2 $c_{ir}$ — 3PL guessing, Beta prior (Béguin-Glas augmentation)

This package uses Béguin-Glas augmentation. So $c_{ir}$ is actually Gibbs, not MH. We show why $c$ is non-conjugate under standard augmentation, then why Béguin-Glas restores conjugacy.

**Standard augmentation approach.** The 3PL likelihood of $Y^{*}$ is:

$$\Pr(Y^{*}_{pir} = 1 \mid \theta, b, a, c) = c_{ir} + (1 - c_{ir})\Phi(a_{ir}(\theta_{pr} - b_{ir})).$$

A Beta prior is conjugate to a Bernoulli likelihood of the form $c^{y}(1-c)^{1-y}$. The likelihood above is not that form. The term $c_{ir} + (1-c_{ir})\Phi_{pir}$ is a mixture of $c$ and $(1-c)$ weighted by $\Phi_{pir}$. That coefficient depends on other parameters. No closed form.

**Béguin-Glas augmentation.** Introduce a binary indicator $U_{pir} \in \{0, 1\}$ for "guessing mode." The generative model is:

$$U_{pir} \mid c_{ir} \sim \mathrm{Bernoulli}(c_{ir}).$$

If $U_{pir} = 1$: the response was a guess; $Y^{*}_{pir} = 1$ automatically.\
If $U_{pir} = 0$: the response came from the ability process; $Y^{*}_{pir} = \mathbb{1}\{Z_{pir} > 0\}$ as usual.

Now $c_{ir}$ only enters through $U_{pir}$. The marginal integrates out to recover the original 3PL model.

**Given** $U$, the likelihood of $c_{ir}$ is Bernoulli with $c$ as the success probability. Beta prior × Bernoulli likelihood = Beta posterior. Conjugate.

**For each item** $i$, count:

$$n_{\text{guess},i} = \sum_{p \in \mathcal{P}_{ir}} U_{pir}, \qquad n_{ir} = |\mathcal{P}_{ir}|.$$

Every relevant cell $(p,i)$ is a Bernoulli$(c_i)$ trial — whether $Y^{*}_{pir}=0$ (where $U_{pir}=0$ deterministically) or $Y^{*}_{pir}=1$ (where $U_{pir}$ is drawn). The Beta-Bernoulli conjugate update adds successes to $\alpha$ and failures to $\beta$:

$$\boxed{c_{ir} \mid U \sim \mathrm{Beta}\!\left(\alpha_{c} + n_{\text{guess},i},\; \beta_{c} + n_{ir} - n_{\text{guess},i}\right).}$$

The second shape $\beta_c + (n_{ir} - n_{\text{guess},i})$ counts all non-guessing trials: all $Y^*=0$ cells (where $U=0$ always) plus all $Y^*=1$ cells where the response came from the ability process ($U=0$).

One Beta draw per item. No accept-reject. Implemented in `gibbs_samplers.gibbs_c_3pl()`. The default prior is $\mathrm{Beta}(5, 17)$ (mean ≈ 0.23). For a `c_shared` group the counts are summed over the member nodes before the single Beta draw (probit 3PL members only; a logit 3PL node in a `c_shared` group is not supported). At a logit 3PL node there is no $U$, and $c$ is updated by MH (§3.4).

**How is** $U_{pir}$ drawn? At each iteration, for each relevant $(p, i)$:

- If $Y^{*}_{pir} = 0$: a correct response is impossible via guessing (guessing always produces $Y^{*}=1$), so $U_{pir}=0$ deterministically. This cell still counts as one non-guessing Bernoulli trial in the $\beta$ update above.
- If $Y^{*}_{pir} = 1$: draw $U_{pir} \sim \mathrm{Bernoulli}(p^{*}_{pir})$ where

$$p^{*}_{pir} = \frac{c_{ir}}{c_{ir} + (1 - c_{ir})\Phi(a_{ir}(\theta_{pr} - b_{ir}))}.$$

This is the posterior probability that the observed correct response came from guessing rather than ability. Implemented in `gibbs_samplers.sample_U_3pl()`.

------------------------------------------------------------------------

### 3.3 $\rho$ — correlation parameter, LKJ prior

**Model.** For $R=2$ nodes: $\theta_{p} \sim \mathcal{N}_{2}(0, \Sigma)$ with $\Sigma = DRD$, $D = \mathrm{diag}(\sigma_{1}, \sigma_{2})$, and $R = \begin{pmatrix}1 & \rho \\ \rho & 1\end{pmatrix}$. Prior:

$$p(\rho \mid \eta) \propto (1 - \rho^{2})^{\eta-1}, \quad \rho \in (-1, 1).$$

$\eta = 1$ is uniform on $(-1,1)$. $\eta > 1$ shrinks toward zero.

We want $p(\rho \mid \theta_{1{:}P}, \sigma_{1}, \sigma_{2})$.

**Step 1.** Log likelihood of $\theta_{1{:}P}$ given $\Sigma$:

$$\log p(\theta_{1{:}P} \mid \Sigma) = -\frac{P}{2}\log|\Sigma| - \frac{1}{2}\sum_{p}\theta_{p}^{\top}\Sigma^{-1}\theta_{p} + \text{const}.$$

**Step 2.** Express in $\rho$. We have $|\Sigma| = \sigma_{1}^{2}\sigma_{2}^{2}(1-\rho^{2})$ and

$$\Sigma^{-1} = \frac{1}{1-\rho^{2}}\begin{pmatrix}1/\sigma_{1}^{2} & -\rho/(\sigma_{1}\sigma_{2}) \\ -\rho/(\sigma_{1}\sigma_{2}) & 1/\sigma_{2}^{2}\end{pmatrix}.$$

So:

$$\theta_{p}^{\top}\Sigma^{-1}\theta_{p} = \frac{1}{1-\rho^{2}}\left(\frac{\theta_{p1}^{2}}{\sigma_{1}^{2}} - \frac{2\rho\theta_{p1}\theta_{p2}}{\sigma_{1}\sigma_{2}} + \frac{\theta_{p2}^{2}}{\sigma_{2}^{2}}\right).$$

**Step 3.** Combine with prior:

$$\log p(\rho \mid \theta, \sigma) = \left(\eta - 1 - \frac{P}{2}\right)\log(1-\rho^{2}) - \frac{1}{2(1-\rho^{2})}\sum_{p}\left(\frac{\theta_{p1}^{2}}{\sigma_{1}^{2}} - \frac{2\rho\theta_{p1}\theta_{p2}}{\sigma_{1}\sigma_{2}} + \frac{\theta_{p2}^{2}}{\sigma_{2}^{2}}\right).$$

**Step 4.** Why no conjugate form. $\rho$ appears both as $\log(1-\rho^{2})$ and inside $1/(1-\rho^{2})$ multiplied by a linear term in $\rho$. That combination is not the kernel of any standard distribution on $(-1, 1)$.

**MH target:**

$$\boxed{\log p(\rho \mid \theta, \sigma) = \left(\eta - 1 - \frac{P}{2}\right)\log(1-\rho^{2}) - \frac{1}{2(1-\rho^{2})}\sum_{p}\left(\frac{\theta_{p1}^{2}}{\sigma_{1}^{2}} - \frac{2\rho\theta_{p1}\theta_{p2}}{\sigma_{1}\sigma_{2}} + \frac{\theta_{p2}^{2}}{\sigma_{2}^{2}}\right).}$$

Proposal: random walk on $z = \mathrm{arctanh}(\rho)$. This maps $(-1,1)$ to $\mathbb{R}$. The Jacobian $|d\rho/dz| = 1-\rho^{2}$ must be included in the MH ratio: it is evaluated at $\rho^*$ and at $\rho$ and does not cancel. Without it $\rho$ is biased toward $\pm 1$. Implemented in `mcmc_sampler._mh_rho_fisher_z()`.

**More than two dimensions.** For a block of $G > 2$ dimensions the correlation matrix is written through its $G(G-1)/2$ partial correlations (the vine/Cholesky parameterization of Lewandowski, Kurowicka and Joe, 2009), under which the LKJ density factorizes as

$$\log p(R) \propto \sum_{k=1}^{G-1}\Bigl(\eta - 1 + \tfrac{G-k-1}{2}\Bigr)\sum_{j} \log(1 - z_{jk}^{2}),$$

and each partial correlation $z_{jk}$ is updated in turn by the same Fisher-$z$ random walk, with the MVN likelihood of all $P$ rows. The chain stores the marginal correlations. Implemented in `mcmc_sampler._mh_correlation_vine()`.

**Where it runs.** Once per θ correlation block (§2.6), and with the same functions for the b- and a-correlation groups (on the $b_i$, or the centred $a_i$, as the rows). A bounded Normal prior on $\rho$ (`NormalPrior(mean=0, variance=..., lower=-1, upper=1)`, density $\propto \exp(-(\rho-m)^2/2v)$ on the bounds) or a shifted Beta (`BetaPrior(alpha, beta, lower=-1, upper=1)`, the Beta density of $(\rho - l)/(u - l)$) is accepted in place of the LKJ term for two members only (neither has a vine form, so $G > 2$ raises `NotImplementedError`). This holds for θ blocks and for b- and a-correlation groups (`_corr_log_prior_fn`).

------------------------------------------------------------------------

### 3.4 Non-augmented nodes: logit 2PL/3PL, GPCM/PCM and custom nodes

A logit node has no Gaussian latent variable (§1.1), and a GPCM node's adjacent-category probabilities have none either, so none of their parameters has a closed-form full conditional. Every parameter they touch is updated by random-walk MH against the node's own likelihood, $\sum_{(p,i): T_r[p,i]=1}\log \Pr(Y^*_{pir} \mid \theta, b, a, c, \text{offset})$, computed by the node's `log_prob`. A custom node is treated the same way with the user's `log_prob`.

**The node models.** Logit 2PL/3PL: $\Pr(Y^*=1) = c_{ir} + (1-c_{ir})\,g(a_{ir}(\theta_{pr} - b_{ir} + \text{offset}_{pir}))$, with $g$ the logistic CDF. GPCM (Muraki, 1992), for $M$ categories with step difficulties $b_{i1}, \ldots, b_{i,M-1}$ (unordered):

$$\Pr(Y^* = j) = \frac{\exp\sum_{k=1}^{j} a_{ir}(\theta_{pr} + \text{offset}_{pir} - b_{ik})}{1 + \sum_{m=1}^{M-1}\exp\sum_{k=1}^{m} a_{ir}(\theta_{pr} + \text{offset}_{pir} - b_{ik})}, \qquad j = 0, \ldots, M-1$$

(the empty sum for $j=0$). PCM is GPCM with $a_{ir}=1$.

**Why the units can be updated in parallel.** Given everything else, the persons' θ on one dimension are conditionally independent (the logit log-likelihood is a sum of row terms), and so are the items' $b$, $a$ and $c$ (column terms). So all units are proposed at once and accepted **one by one**, each on its own log-ratio. This is exact Metropolis-within-Gibbs over many independent blocks, vectorized.

**The targets.** Each unit's target is a Gaussian part (its prior, or its conditional prior under a correlation, plus the latent terms of any probit node sharing the parameter, §3.5) plus the logit log-likelihood of the unit's cells:

$$\log \pi(v_u) = \ell_u^\top v_u - \tfrac12 v_u^\top \Lambda_u v_u + \sum_{r \text{ logit}}\;\sum_{\text{cells of } u}\log \Pr(Y^*_{pir} \mid v_u, \ldots).$$

| Block | Unit | Working scale and Jacobian | Helper |
|---|---|---|---|
| θ on a dimension with a logit/GPCM node | person | θ (none); prior is the conditional $\theta_{pd} \mid \theta_{p,-d}$ from $\Sigma^{-1}$; every node on the dimension enters | `_mh_diag_units` |
| $b$ of a group with a logit/GPCM member; GPCM steps | item ($k = M-1$ steps at once for GPCM) | $b$ (none) | `_mh_diag_units` |
| $a$ at a logit/GPCM node | item (one for the whole node with `common_discrimination`) | $\xi = \log a$, $+\log a$ | `_mh_logit_discrimination` |
| $c$ at a logit 3PL node | item | $\zeta = \mathrm{logit}\,c$, $+\log c(1-c)$ | `_mh_logit_guessing` |
| β, κ, γ of a group with a logit/GPCM member | the group's vector | $c$ (none); quadratic part = prior + probit latent terms | `_mh_coefficient` |
| missing $X$, $Q$, $W$ used by a logit/GPCM node | person / item / cell | value (none); Gaussian part from §2.11.2 | `_mh_missing_units` |

**Proposals.** θ and $b$ proposals are $v' = v + s\,\epsilon/\sqrt{H_u}$, with $H_u$ the Gaussian precision plus a bound on the logit information, so one step size fits units with very different amounts of data. Coefficient and missing-value proposals use a fixed Cholesky factor of an approximate posterior covariance (the Gaussian precision plus ¼ of the logit information, the logistic's maximum curvature); because it does not depend on the current value, the proposal stays symmetric. Step sizes are tuned during warm-up toward the optimal acceptance for the block's dimension (Roberts, Gelman and Gilks, 1997: ≈0.44 for one parameter, ≈0.23 for many) and then frozen.

### 3.5 Groups that mix probit and logit members

A parameter can be shared by a probit node and a logit node (a shared trait, a shared $b$ or $a$ group, a pooled coefficient). Its full conditional is then the product of both nodes' likelihoods with the prior. Given $Z$ at the probit members, their part is the exact Gaussian term of §2 (it does not need to be sampled around); the logit members' part is their log-likelihood. So the target is

$$\log \pi(v) = \underbrace{\ell^\top v - \tfrac12 v^\top\Lambda v}_{\text{prior + probit members' latent terms}} + \underbrace{\sum_{r\ \text{logit}} \log p(Y^*_r \mid v, \ldots)}_{\text{logit members}},$$

updated by the MH helpers of §3.4 with every member in the target. This is one block for the shared parameter, never one update per member node (which would discard all members but the last, §2.11.4). For a shared slope array with a logit member, for example, each item's proposal on $\log a_i$ is accepted against the probit members' $-\tfrac12\sum(Z - a_i d)^2$ plus the logit members' column log-likelihood plus the prior and the Jacobian.

------------------------------------------------------------------------

## 4. Note on $\sigma^{2}_{r}$ under a HalfNormal prior

The v0.1 package used a HalfNormal(0, 1) prior on the SD $\sigma_r$ (equivalently, on $\sigma^2_r$, the density below). This is a sensible prior (Gelman, 2006). It places mass on small values and avoids the heavy right tail of InvGamma. So why was it changed?

**The conjugacy failure.** With $\sigma \sim \mathrm{HalfNormal}(0, s)$, the implied density on $\sigma^{2}$ is ($s = 1$ here)

$$p(\sigma^{2}) = p_\sigma(\sqrt{\sigma^2})\,\Bigl|\frac{d\sigma}{d\sigma^2}\Bigr| \propto \frac{1}{\sqrt{\sigma^{2}}}\exp\!\left(-\frac{\sigma^{2}}{2s^2}\right)\mathbb{1}\{\sigma^{2} > 0\}.$$

The Gaussian likelihood for $\theta_{1{:}P}$ given $\sigma^{2}$ contributes two types of terms to $\log p(\sigma^{2} \mid \theta)$:

- A term in $\log\sigma^{2}$, from the normalizer $(2\pi\sigma^{2})^{-P/2}$.
- A term in $1/\sigma^{2}$, from the quadratic form $\sum\theta_{p}^{2}/(2\sigma^{2})$.

The HalfNormal prior adds a third type: a term in $\sigma^{2}$ itself, from $\exp(-\sigma^{2}/2)$. So the full conditional has terms in $\log\sigma^{2}$, $1/\sigma^{2}$, and $\sigma^{2}$ simultaneously. That combination matches no standard distribution on $(0, \infty)$.

**Why InvGamma works.** Its kernel is $(\sigma^{2})^{-\alpha-1}\exp(-\beta/\sigma^{2})$. This has terms in $\log\sigma^{2}$ and $1/\sigma^{2}$ only. Exactly what the Gaussian likelihood produces. The InvGamma prior plays the same conjugate role for $\sigma^{2}$ that the Normal prior plays for $\theta$: the kernel shapes match, so they combine cleanly.

**The practical cost of HalfNormal.** Every $\sigma^{2}_{r}$ update requires an MH step with Robbins-Monro adaptation. For a model with several nodes, this adds several MH blocks to every iteration, each needing step-size tuning and each adding autocorrelation. Switching to InvGamma eliminates all of them. That is the entire reason for the change.

**When HalfNormal is still acceptable.** If you have strong prior knowledge about the *scale* of $\sigma_{r}$ — for example, "the trait SD is unlikely to exceed 3" — HalfNormal expresses this more naturally. The MH cost is small per step; it just adds up over many iterations and many nodes. The opt-in path in the package emits a UserWarning at construction so the user is aware.

**The HalfNormal MH target** (for the opt-in case). In $\sigma^2$,

$$\log p(\sigma^{2} \mid \theta) = -\frac{P+1}{2}\log\sigma^{2} - \frac{\sum\theta_{p}^{2}}{2\sigma^{2}} - \frac{\sigma^{2}}{2s^2} + \text{const}.$$

The chain moves in $z = \log\sigma$ with a symmetric random walk, so the target is the $\sigma$-scale posterior times the Jacobian $|d\sigma/dz| = \sigma$:

$$\boxed{\log \pi(z) = -P z - \frac{\sum\theta_p^2}{2e^{2z}} - \frac{e^{2z}}{2s^2} + z.}$$

Without the final $+z$ the chain targets the posterior times $1/\sigma$ and biases $\sigma$ toward 0. Implemented in `mcmc_sampler._mh_sigma2_halfnormal()`.

------------------------------------------------------------------------

## 4a. Hierarchical variances and the conjugacy boundary

Any prior's variance slot may hold a `PriorSpec` on $(0,\infty)$ instead of a
number, promoting that variance to an estimated hyperparameter. Whether the
estimate is Gibbs or MH is decided by one question: **is the parameter's own prior
an unbounded Normal?**

**Gibbs case — unbounded Normal parameter ($\theta$, $b$, $\beta$, $\kappa$,
$\gamma$).** The variance's full conditional is the InvGamma of §1.3 / §2.7 / §2.8,
with the parameter's drawn values as data. All five share one implementation,
`gibbs_variance_invgamma`, aliased as `gibbs_sigma2` (θ) and `gibbs_b_variance`
(b/β/κ/γ). The sufficient statistic is $\sum(x-\mu)^2$ about the prior mean $\mu$;
for the zero-centred priors used here $\mu=0$, but the implementation carries $\mu$
explicitly so a non-zero-centred prior is not silently biased.

**MH case — bounded parameter (discrimination $a$, `TruncNormalPrior`).** Here the
InvGamma posterior does **not** close, for the same reason the HalfNormal case in §4
does not: the truncated-Normal prior on $a$ carries a normalising constant that
depends on $\sigma$. Writing the prior on the $I$ discriminations $a_{1:I} \sim
\mathcal{N}(\mu_a, \sigma^2)$ truncated to $(l, u)$ (here $l=0$, $u=\infty$),

$$\log p(\sigma^2 \mid a) = -\tfrac{I}{2}\log\sigma^2 - \frac{\sum_i (a_i-\mu_a)^2}{2\sigma^2} - I\log\!\Bigl[\Phi\bigl(\tfrac{u-\mu_a}{\sigma}\bigr) - \Phi\bigl(\tfrac{l-\mu_a}{\sigma}\bigr)\Bigr] + \log p_{\text{IG}}(\sigma^2) + \text{const}.$$

The first two terms are the InvGamma-conjugate part; the third — the truncation
normaliser $-I\log[\Phi(\cdot)-\Phi(\cdot)]$ — is a function of $\sigma$ that no
InvGamma can absorb. Dropping it (treating the draw as conjugate) biases $\sigma^2$
upward, because a wider $\sigma$ spills prior mass outside $(l,u)$ and the missing
renormalisation never penalises it.

$$\boxed{\text{MH target: } \log p(\sigma^2\mid a) \text{ above, with a random walk on } \log\sigma^2 \text{ (Jacobian } +\log\sigma^2\text{).}}$$

**Enforcement.** `priors._check_variance_conjugacy` raises `TypeError` if a
`PriorSpec` is placed in the variance slot of a *bounded* prior, unless the user
passes `force=True`, which downgrades to a `UserWarning` and activates the MH block
(`AdaptState` key `a.variance_{g}`, Robbins-Monro tuned). The unbounded-Normal
variances take the Gibbs path with no warning.

**Other variances estimated by MH.** (i) The variances of a-correlation groups under an
LKJ or fixed `a.correlation`, estimated by default with a Half-Normal(0, 0.5) prior on
the SD (§2.13); their MVN likelihood ignores the orthant normalizer. (ii) Any variance
whose hyperprior is a `HalfNormalPrior` (§4).

------------------------------------------------------------------------

## 5. Moves that improve mixing

Each update in §2–§3 is a valid step on its own. Three extra moves exist because some directions of the posterior are crossed very slowly by one-block-at-a-time updates. Each leaves the target invariant, so adding it never changes what the chain converges to, only how fast.

### 5.1 Cowles step for ordered thresholds

**The problem.** Given $Z$, a GRM threshold can only move inside $(L_k, U_k)$ (§2.10), the gap between the largest latent response of category $k$ and the smallest of category $k+1$. With hundreds of respondents per item that gap is tiny, so the exact draw barely moves and the chain for $\tau$ (and for $Z$, which is pinned by $\tau$) crawls.

**The move** (Cowles, 1996). Integrate $Z$ out. The marginal likelihood of item $i$'s thresholds is the product of the model's category probabilities over every node sharing them, e.g. $\Phi(\tau_{i,m} - \mu_{pi}) - \Phi(\tau_{i,m-1} - \mu_{pi})$ with $\mu_{pi} = \theta_{pr} + \text{offset}_{pir}$ at a GRM node. Propose all thresholds of the item jointly by a symmetric random walk on

$$u = (\tau_{i,0},\ \log(\tau_{i,1} - \tau_{i,0}),\ \ldots,\ \log(\tau_{i,M-2} - \tau_{i,M-3})),$$

so every proposal is ordered. The target in $u$ is

$$\log\pi(u) = \sum_{r}\sum_{p}\log\Pr(Y^*_{pir} \mid \tau(u), \mu) - \frac{1}{2\tau^2}\sum_k(\tau_{i,k} - \mu_0)^2 + \sum_{k\ge 1} u_k,$$

the last term being the Jacobian $\prod_k(\tau_{i,k} - \tau_{i,k-1})$. Then redraw $Z$ given the new thresholds and make the exact draw of §2.10. The pair (Cowles step, $Z$ redraw) is a joint update of $(\tau_i, Z_{\cdot i})$, so it is not confined to the gap.

Items are conditionally independent, so all are proposed together and accepted one by one. Per-coordinate proposal scales are fixed from the data alone (cumulative category proportions and a delta-method SD, `_threshold_proposal_scale`), which keeps the proposal symmetric; one step size per group is tuned during warm-up. Implemented in `mcmc_sampler._mh_ordered_thresholds()`.

### 5.2 Location move along the θ/b ridge

**The problem.** Shifting $\theta_{pd}$ for every person and $b_{ir}$ (and every threshold) for every item on dimension $d$ by one constant $c$ leaves every likelihood term unchanged: $\theta - b$, $Z - \tau$ and $Z - \theta$ are invariant, and binary $Z \sim a(\theta - b)$ is untouched. Only the priors depend on $c$. Alternating θ and $b$ draws can cross this direction only in small steps when the data are informative.

**The move** (Liu and Sabatti, 2000, a Gibbs step on a translation group). Draw $c$ from its exact conditional and apply the shift to θ on $d$, every $b$ of the b-groups on $d$, and the GRM latent responses. The log conditional of $c$ is the sum of the prior terms:

$$\log p(c) = -\frac{\Lambda_{dd}}{2}\sum_p(\theta_{pd} + c - m_{pd})^2 - \sum_{g}\frac{1}{2\tau_g^2}\sum_i(b_{ig} + c - \mu_{b,g})^2 + \text{const},$$

with $\Lambda = \Sigma^{-1}$ and $m_{pd} = -\sum_{e\ne d}\Lambda_{de}\theta_{pe}/\Lambda_{dd}$ the conditional prior mean given the other dimensions. It is quadratic in $c$, so

$$\boxed{c \sim \mathcal{N}\!\left(\frac{-\Lambda_{dd}\sum_p(\theta_{pd} - m_{pd}) - \sum_g\sum_i(b_{ig} - \mu_{b,g})/\tau_g^2}{P\Lambda_{dd} + \sum_g I_g/\tau_g^2},\; \frac{1}{P\Lambda_{dd} + \sum_g I_g/\tau_g^2}\right).}$$

It is applied to a dimension when every member of each b-group used on it is on that dimension and none of those groups is in a b-correlation group (whose prior would couple the shift to other groups). Implemented inline in `run_chain` after the θ draw.

### 5.3 Interweaving for covariate effects (ASIS)

**The problem.** The θ, $b$ and coefficient draws use the *non-centred* form: $b_i \sim \mathcal{N}(\mu_b, \tau^2)$ with predictor $\theta - b_i + Q_i^\top\kappa$. In that form κ can move only as far as $b$ lets it, so the pair crawls along the ridge $b - Q\kappa = \text{const}$ (§2.11.5). The same holds for β and θ.

**The move** (Yu and Meng, 2011). In the *centred* form the location $\eta_i = b_i - Q_i^\top\kappa$ carries the likelihood and κ enters only its prior: $\eta_i \sim \mathcal{N}(\mu_b - Q_i^\top\kappa, \tau^2)$. When every node sharing $b$ uses this κ group, the likelihood depends on $(b, \kappa)$ only through η, so

$$\eta = b - Q\kappa; \qquad \kappa \sim p(\kappa \mid \eta); \qquad b = \eta + Q\kappa$$

is an exact Gibbs step in the centred parameterization, and it leaves every likelihood term (and every cached logit likelihood) unchanged. $p(\kappa \mid \eta)$ is a Bayesian linear regression, $\mu_b - \eta_i = Q_i^\top\kappa + e_i$ with $e_i \sim \mathcal{N}(0, \tau^2)$, under κ's Normal prior. For β the centred location is $\zeta_{pd} = \theta_{pd} + X_p^\top\beta$, and the regression uses the conditional prior of $\theta_{pd}$ given the other dimensions (precision $\Lambda_{dd}$). Interleaving the non-centred draws (§2.11.1) with this centred one combines the strengths of both: whichever form mixes well for the data at hand dominates. Implemented inline in `run_chain` (`_asis_draw`), after the coefficient draws; skipped for a group when some node sharing $b$ (or θ) does not use the coefficient, or when $b$ is in a b-correlation group or is mixed-format.

## 6. Summary table

Defaults are marked (default). "Gibbs" means an exact draw from the full conditional; "MH" a Metropolis-Hastings step tuned during warm-up.

| Parameter | Prior | Full conditional | Update | § |
|----|----|----|----|----|
| $\theta_{p}$, all probit nodes (Rasch, 2PL/3PL, GRM; independent or correlated) | $\mathcal{N}_D(0, \Sigma)$, $\Sigma = DRD$ | (Multivariate) Normal, weighted by $a^2$; no $b$ term at GRM nodes | Gibbs | 2.1–2.6 |
| $\theta$ on a dimension with a logit/GPCM/custom node | same | none | MH per person | 3.4 |
| $b_{ir}$, probit nodes | $\mathcal{N}(\mu_b, \tau^2)$ (default $\mu_b=0$) | Normal (weighted) | Gibbs | 2.2, 2.4 |
| $b$ of a group with a logit/GPCM member; GPCM steps | same | none | MH per item | 3.4–3.5 |
| $\tau_{i,k}$ (GRM thresholds) | $\mathcal{N}(\mu_0, \tau^2)$, ordered | prior truncated to $(L_k, U_k)$, after a Cowles step | Gibbs (+ MH move) | 2.10, 5.1 |
| $\sigma^{2}_{d}$ | InvGamma(1, 1) on dimensions with a slope-free node; fixed at 1 where every node has free slopes (default) | Inverse-Gamma | Gibbs | 1.4, 2.7 |
| $\sigma_{d}$ (opt-in) | HalfNormal$(0, s)$ on the SD | none | MH on $\log\sigma$ | 4 |
| $\tau^{2}_{g}$ (per b-group), and β/κ/γ variances | InvGamma(1, 1) (default) | Inverse-Gamma | Gibbs | 2.8, 4a |
| $a_{ir}$, probit 2PL/3PL | TruncNormal$(1, 0.5)$ on $(0,\infty)$ (default) | Truncated Normal | Gibbs | 3.1a |
| $a_{r}$, one slope per node (`common_discrimination`) | same | Truncated Normal, pooled over all cells | Gibbs | 3.1a |
| $a_{ir}$, probit, LogNormal prior | LogNormal$(\mu_a, \sigma_a^2)$ | none | MH on $\log a$ | 3.1 |
| $a_{ir}$, logit/GPCM | either | none | MH on $\log a$ per item | 3.4 |
| $a$-group variance, uncorrelated (`force=True`) | variance slot of TruncNormal $a$ | none (truncation normalizer) | MH on $\log v$ | 4a |
| $c_{ir}$, probit 3PL (Béguin–Glas) | Beta(5, 17) (default) | Beta | Gibbs | 3.2 |
| $c_{ir}$, logit 3PL | same | none | MH on logit $c$ | 3.4 |
| $\rho$ / $R$ (θ blocks, b- and a-correlation) | LKJ$(\eta)$, or bounded Normal ($G=2$) | none | MH on Fisher-$z$ (vine for $G>2$) | 3.3 |
| $\Sigma_\theta$ (per block), $\Sigma_b$, $\Sigma_a$ | InvWishart$(\Psi_0, \nu_0)$ (opt-in) | Inverse-Wishart | Gibbs | 2.14 |
| $b_{i}$ across a b-correlation group | $\mathcal{N}_{G}(\mathbf{0}, \Sigma_{b})$ | Multivariate Normal (probit groups) | Gibbs; MH for logit groups | 2.12 |
| $a_{i}$ across an a-correlation group | $\mathcal{N}_{G}(\mu_a, \Sigma_a)$ on $(0,\infty)^G$ | Truncated Normal per component | Gibbs (componentwise); MH for logit groups | 2.13 |
| a-correlation group variances | Half-Normal(0, 0.5) on the SD (default under LKJ/fixed) | none | MH on $\log v$ | 2.13 |
| β, κ, γ (one vector per group) | $\mathcal{N}(\mu_0, \tau^2_g I)$ | Multivariate Normal (pooled over member nodes) | Gibbs; MH with a logit member | 2.11.1, 3.4 |
| missing $X$, $Q$, $W$ features | $\mathcal{N}(0, 1)$ per feature (default) | Multivariate Normal, given observed features | Gibbs; MH for units a logit node uses | 2.11.2, 3.4 |
| common θ/b location | (priors of θ and b) | Normal | Gibbs move | 5.2 |
| κ given $b - Q\kappa$; β given $\theta + X\beta$ | (coefficient prior) | Multivariate Normal | Gibbs move (ASIS) | 5.3 |

**Note on the table.** Every "weighted" and covariate row carries the discrimination weight at 2PL/3PL nodes: precisions gain $a_{ir}^2$ and linear statistics gain $a_{ir}$ (design rows become $a_{ir}\times$covariate). The formulas in §2 are written for one node; shared parameters pool over member nodes by adding precisions and linear statistics, with the prior counted once.

------------------------------------------------------------------------

## References

- Albert, J. H., & Chib, S. (1993). Bayesian analysis of binary and polychotomous response data. *Journal of the American Statistical Association*, 88(422), 669–679.
- Béguin, A. A., & Glas, C. A. W. (2001). MCMC estimation and some model-fit analysis of multidimensional IRT models. *Psychometrika*, 66(4), 541–561.
- Cowles, M. K. (1996). Accelerating Monte Carlo Markov chain convergence for cumulative-link generalized linear models. *Statistics and Computing*, 6(2), 101–111.
- Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models. *Bayesian Analysis*, 1(3), 515–534.
- Haley, D. C. (1952). *Estimation of the dosage mortality relationship when the dose is subject to error* (Technical Report No. 15). Applied Mathematics and Statistics Laboratory, Stanford University.
- Johnson, V. E., & Albert, J. H. (1999). *Ordinal Data Modeling*. Springer, Ch. 4.
- Klein Entink, R. H., Fox, J.-P., & van der Linden, W. J. (2009). A multivariate multilevel approach to the modeling of accuracy and speed of test takers. *Psychometrika*, 74(1), 21–48. (Conjugate Normal-InverseWishart prior for item parameters; foundation for InverseWishartPrior and §2.14.)
- Lewandowski, D., Kurowicka, D., & Joe, H. (2009). Generating random correlation matrices based on vines and extended onion method. *Journal of Multivariate Analysis*, 100(9), 1989–2001.
- Liu, J. S., & Sabatti, C. (2000). Generalised Gibbs sampler and multigrid Monte Carlo for Bayesian computation. *Biometrika*, 87(2), 353–369.
- Liu, J. S., Wong, W. H., & Kong, A. (1994). Covariance structure of the Gibbs sampler with applications to the comparisons of estimators and augmentation schemes. *Biometrika*, 81(1), 27–40.
- Muraki, E. (1992). A generalized partial credit model: Application of an EM algorithm. *Applied Psychological Measurement*, 16(2), 159–176.
- Patz, R. J., & Junker, B. W. (1999). A straightforward approach to MCMC methods for item response models. *Journal of Educational and Behavioral Statistics*, 24(2), 146–178.
- Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *The Annals of Applied Probability*, 7(1), 110–120.
- Sahu, S. K. (2002). Bayesian estimation and model choice in item response models. *Journal of Statistical Computation and Simulation*, 72(3), 217–232.
- Samejima, F. (1969). Estimation of latent ability using a response pattern of graded scores. *Psychometrika Monograph Supplement*, 34(4, Pt. 2), No. 17.
- Yu, Y., & Meng, X.-L. (2011). To center or not to center: That is not the question. An ancillarity–sufficiency interweaving strategy (ASIS) for boosting MCMC efficiency. *Journal of Computational and Graphical Statistics*, 20(3), 531–570.
- Zellner, A. (1962). An efficient method of estimating seemingly unrelated regressions and tests for aggregation bias. *Journal of the American Statistical Association*, 57(298), 348–368. (Normal linear regression conjugacy; the template used for β, κ, γ and the missing covariates.)
