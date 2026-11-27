# Variational Difficulty Profile (VDP) v0.1

Status: proposed specification for Research Task 002; review before implementation. Date: 2026-11-27. Supplements RESEARCH.md, AGENTS.md and ROADMAP.md. Preserve Task 001.

## 1. Scientific objective

Can inexpensive MFVI provide actionable evidence about inference difficulty, enabling selection of the simplest reliable algorithm under a hard computational budget?

The profile is a structured vector, not five arbitrary scalar scores:

$$
\widehat{\mathbf C}=(C_{\rm opt},C_{\rm dep},C_{\rm tail},C_{\rm geom},C_{\rm mode}).
$$

Each component contains measurements, uncertainty, per-run values, status, and cost. Difficulty depends on parameterization, objective, family, optimizer, initialization and budget; it is not intrinsic to the density. C_mode describes evidence of distinct variational solutions, not the number of posterior modes. No universal quality thresholds or learned decision policy are assumed.

## 2. Target access

In unconstrained coordinates, p(x)=tilde p(x)/Z, ell(x)=log tilde p(x), g=∇ell, H=-∇²ell. Z is unknown. Assume a proper target, compatible support and sufficient differentiability/integrability for each requested estimator. Include the log absolute Jacobian of any declared coordinate transformation. Flag support, derivative and numerical failures instead of silently modifying the target.

Inference receives dimension, optional user-supplied reference location/scale, and metered ell, gradient and Hessian-vector products (HVPs). It can sample/evaluate q. It cannot access target samples, true moments, covariance, precision, optima, normalizer, mode labels or revealing target metadata. Wrap Gaussian targets so their implementation details are not exposed. Independent evaluator alone receives oracle data. Task 002 uses trusted code with tested restricted interface; adversarial autonomy would require actual process isolation. Use float64 and log-space arithmetic. Relative diagnostics are invariant to ell+c; absolute ELBO and log weights shift by c.

## 3. Hard computational budget

Primary raw count N_f = individual target log-density evaluations; separately N_g = gradients, N_h = point-direction HVPs. Proposed v0.1 charge convention: one target-evaluation-equivalent unit (TEU) per counted primitive. This is not a claim of equal runtime.

| Interface | Charge per point | Counter increments |
|---|---:|---|
| value(x) | 1 | f=1 |
| value_grad(x) | 2 | f=1,g=1 |
| value_grad_hvp(x,v) | 3 | f=1,g=1,h=1 |

The HVP interface returns all three quantities even if only HVP is used. Charge full triple for repeated directions; no caching discounts or direct gradient API in v0.1. Fused AD operations have same logical charges. Batch m costs m times the per-point charge. Hard invariant C=N_f+N_g+N_h<=B. Atomically reserve complete charges before executing; deny unaffordable operations with no partial execution or optimizer mutation. No refund after attempted failure; distinguish reserved, completed, failed, denied requests. Reject invalid charges. All target access through meter; no direct backward through raw target. Account for initialization, four runs, diagnostics, restarts, adaptation and deployable validation. Evaluator-only costs separate. Also record actual time, memory, backend, device and dimension. Changes to charge weights require separately frozen protocol.

Reserve diagnostic capacity before equal-horizon training. Budget stop occurs before unaffordable next operation (remaining need not be zero); preserve last valid endpoint and unresolved warnings. Failure does not permit unplanned reallocation or replacement runs. Budget exhaustion is not convergence. Future early reliability stopping needs calibrated gates; v0.1 uses fixed horizons.

## 4. Variational probe

q=N(mu,D), D=diag(sigma²), lambda=log sigma, x=mu+sigma⊙epsilon, epsilon~N(0,I). Unnormalized ELBO:

$$
L=E_\epsilon[\ell(x)]+\sum_i\lambda_i+\tfrac d2(1+\log(2\pi))=\log Z-KL(q\|p).
$$

Pathwise ascent gradients, under appropriate integrability:

$$
\nabla_\mu L=E[g(x)],\qquad\nabla_\lambda L=\mathbf1+E[g(x)\odot\sigma\odot\epsilon].
$$

Use analytic entropy and explicit sample gradients from detached metered value_grad; Adam minimizes -L. Verify equivalence to original reparameterized gradient with identical epsilon in unclamped interior. Logged sampled objective may differ from Task 001. Default Adam lr=.03, betas (.9,.999), eps=1e-8, no weight decay, constant lr, CPU float64. Retain Task 001 clamp [-12,8], match its derivative convention, log raw and effective lambda and saturation. Stationarity diagnostics refer to effective parameters, not clipped optimizer stationarity.

Four runs, root IDs 0,1,2,3; derive independent initialization, training, gradient-check, importance, curvature and bootstrap RNG streams via versioned stable seed derivation (e.g. SeedSequence with fixed integer tags). Record realized seeds. Pair training streams across horizons, not independent replications. Default reference b=0, a=1 (or frozen user-declared b and positive scales a). Independent starts mu=b+a⊙N(0,I), lambda=log a+.25 N(0,I), subject to clamp. Common-start control mu=b, lambda=log a with independent training draws isolates gradient randomness. Four starts cannot rule out unseen modes.

### 4.1 Proposed development protocol

m=64 samples per step, T in {250,500,1000,3000}. Log initial state and every post-update raw/effective parameter, learning rate, pre-update training objective, gradient and charges. Logging requires no extra target calls. Per completed run reserve G=256 fresh gradient-check samples, S=1024 iid importance draws, J=8 curvature locations (first J importance draws), R=16 iid Rademacher directions shared across those locations. Budget:

- Training: 8Tm TEU across four runs.
- Stationarity: 8G TEU.
- Importance/ELBO: 4S TEU.
- HVP: 12JR TEU (geometry reuses dependence HVPs).
- Trajectory and pairwise KL: zero target calls.

Diagnostic reservation 7,680 TEU; total B=135,680; 263,680; 519,680; 1,543,680 respectively. Proposed engineering defaults, not validated choices. Unsupported HVP leaves its reservation unused. Freeze all settings before execution. Fresh independent diagnostics at each horizon, separately charged; do not reuse uncharged intermediate checkpoints.

## 5. Optimization and agreement

Use final 50% of post-update records, aligned equal-length windows, fixed lr. Require at least 40 original records per run (20 per split); engineering minimum. Split dropping middle observation if odd. Evaluate every mu and effective lambda coordinate.

For M chains of n observations, means a_m, pooled mean a, sample variances s_m²,

$$
W=M^{-1}\sum_m s_m^2,\quad B_R=\frac n{M-1}\sum_m(a_m-a)^2,\quad
\hat R=\sqrt{[(n-1)W/n+B_R/n]/W}.
$$

Report classical unsplit M=4 and split M=8. For rank-normalized split R-hat pool N=8n values, midranks r, normal scores Phi^{-1}((r-3/8)/(N+1/4)); apply formula. Fold raw split values around pooled median, rank-normalize absolute deviations, repeat. Report rank and folded components and maximum, per-coordinate values, median, max, quantiles, fraction above 1.01 (MCMC reference only). If W=B_R=0 mark constant/undefined; W=0,B_R>0 mark separated-constant/infinite (null plus status). No epsilon denominator or selective omission. Missing runs invalidate four-run metric. Compare to pinned reference implementation. Small within-run variance can inflate R-hat; Adam state matters, iterates are not posterior draws. Unimodality does not guarantee unique variational optimum.

### 5.1 Drift and independent gradient

Split final window into equal early/late halves, means mu1,mu2,lambda1,lambda2. Scale s_ref=exp((lambda1+lambda2)/2). Report RMS and max absolute coordinate drift of (mu2-mu1)/s_ref and lambda2-lambda1, signed coordinate differences. Descriptive, not iid inference.

At final effective phi draw G fresh epsilons. Each standardized ascent gradient h=(sigma⊙g, 1+sigma⊙epsilon⊙g). Report mean vector, sample coordinate SE=sd/sqrt(G) if finite variance plausible, RMS norm ||mean h||/sqrt(2d), noise RMS=sqrt(sum(SE²)/(2d)), maximum absolute coordinate mean, clamp flags. No simultaneous significance test or guaranteed confidence coverage. High uncertainty, saturation or nonfinite gradients imply inconclusive stationarity; small gradient does not certify global optimum.

### 5.2 Fitted-distribution agreement

For all six unordered pairs of final diagonal Gaussians:

$$
KL(q_a\|q_b)=\tfrac12\sum_i[\log(v_{bi}/v_{ai})+(v_{ai}+(\mu_{ai}-\mu_{bi})^2)/v_{bi}-1].
$$

J_ab=(KL(q_a||q_b)+KL(q_b||q_a))/2. Report matrix, six values, median/max, joint nats and nats/dimension, mean/scale contributions, coordinate mean differences standardized by sqrt((v_a+v_b)/2), log-scale differences. Stable log-ratio algebra. Missing fits yield missing pairs. Costs O(d) per pair and zero target calls. Parameter nonidentifiability matters in future richer families; compare distributions rather than labels.
