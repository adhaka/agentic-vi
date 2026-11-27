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

## 6. Dependence: standardized curvature

Let A(x)=D^{1/2}H(x)D^{1/2}, using each run's final D. For a Gaussian, H=P=Sigma^{-1}; at reverse-KL MFVI optimum v_i=1/P_ii. Define the local interaction statistic

$$
c_{\rm dep}(x)=\frac{\|A-\mathrm{diag}(A)\|_F}{\|A\|_F}.
$$

For a standard bivariate Gaussian with correlation rho, optimal A=[[1,-rho],[-rho,1]], c_dep=|rho|/sqrt(1+rho²), marginal variance recovery=1-rho². This is a local curvature measure, not marginal correlation or an objective-independent family error. A diagonal local Hessian cannot exclude global nonlinear dependence. Indefinite H is permitted. Consistent diagonal affine coordinate changes preserve the statistic (up to sign/permutation); general rotations and nonlinear transforms need not.

### 6.1 Matrix-free estimation

Draw J iid locations from each final q; reuse first J of the S independent importance draws. Draw R independent Rademacher vectors z_r, shared across locations within a run, independent of locations. Compute u_jr=A(x_j)z_r by applying the negative log-density HVP in direction D^{1/2}z_r, then left scaling by D^{1/2}. Set t_jri=z_ri u_jri. Conditional on a location and D,

$$
E_z[t_{jri}]=A_{j,ii},\quad \mathrm{Var}_z(t_{jri})=\sum_{k\ne i}A_{j,ik}^2.
$$

Estimate total and off-diagonal squared Frobenius energy:

$$
\widehat F_j=\frac1R\sum_r\|u_{jr}\|^2,\qquad
\widehat O_j=\sum_i\frac1{R-1}\sum_r(t_{jri}-\bar t_{ji})^2.
$$

Both energy estimators are unbiased conditional on location and D; R>=2. The primary run-level ratio is

$$
\widehat C_{\rm dep}=\sqrt{\frac{\sum_j\widehat O_j}{\sum_j\widehat F_j}}.
$$

It estimates an energy-weighted interaction over q, not the mean of local ratios, and is not unbiased. Report numerator and denominator, local ratios, estimated diagonals, total energy/dimension, extrema. Finite-probe noise may yield O_hat>F_hat; preserve raw ratio and flag, never silently clip. Zero denominator means undefined; nonfinite results invalid. Cost JR HVPs per run, O(JRd) stored vectors or streaming summaries, no dense d-by-d matrices. Dense oracle permitted only for independent small-dimensional tests.

### 6.2 Uncertainty

Use two-way bootstrap sensitivity: independently resample J location indices and R direction indices with replacement; share resampled direction indices across selected locations. Recompute statistic, 200 replicates using separate fixed RNG. Report 2.5/97.5 percentiles and invalid replicate count. This is exploratory Monte Carlo sensitivity, not validated confidence coverage at small J,R. Constant curvature location resampling adds no variation. Retain raw per-run estimates separately; four runs do not characterize population uncertainty. No extra target calls for bootstrap.

## 7. Geometry: curvature variation at zero extra target cost

With the same D, locations, directions and HVPs, define u_bar_r=mean_j u_jr.

$$
\widehat V=\frac1{R(J-1)}\sum_{r,j}\|u_{jr}-\bar u_r\|^2,\qquad
\widehat F=\frac1{JR}\sum_{j,r}\|u_{jr}\|^2,\qquad
\widehat C_{\rm geom}=\sqrt{\widehat V/\widehat F}.
$$

For J>=2 the numerator estimates E_q||A-E_q A||_F², denominator estimates E_q||A||_F² (finite second moments assumed). Ratio not unbiased; can exceed one at finite J. Report raw energies/ratio, flags, two-way bootstrap sensitivity as above. Undefined for zero denominator. Gaussian constant H implies V=0 up to numerical tolerance, regardless of correlation. Report sampled directional curvatures zᵀu/||z||², minimum and negative fraction. Negative directional curvature is local evidence of nonconvexity; no negative probes does not certify convexity. Samples from q may not explore problematic regions. More extensive geometry probes, finite difference HVPs or spectral estimates deferred to separately charged study.

## 8. Tail/coverage: importance weights and independent ELBO

For each final q independently draw S iid new samples, separate from training and gradient checks. Compute a_s=ell(x_s)-log q(x_s); w_s=exp(a_s-logsumexp(a)). Then

$$
ESS_{\rm raw}=\frac1{\sum_s w_s^2},\qquad ESS_{\rm relative}=ESS_{\rm raw}/S.
$$

Report both ESS measures and max weight. Use version-pinned PSIS reference implementation, reff=1 for iid proposal draws. Report Pareto shape k_hat, smoothed normalized log weights and ESS_psis=1/sum(w_psis²), package version and tail-fit warnings. These weight ESS values are not MCMC ESS or actual independent posterior sample sizes. Use documented tail-fitting procedure, verify reference fixtures. The sample-size-dependent reference k_ref=min(1-1/log10(S),0.7) is an importance-sampling reliability warning, not a posterior-accuracy gate. Constant finite log weights imply raw ESS=S; if tail fit is degenerate mark k_hat undefined/constant, never invent a perfect estimate. NaN/+inf invalid; -inf weights indicate zero target density and potential support/ELBO failure (record counts), all-zero weights invalid. Do not drop difficult draws or select only favorable runs. Repeated independent batches for k uncertainty require separate budget; otherwise report no interval and finite-sample limitation.

Report independent final ELBO L_hat=mean(a), SE=sd(a)/sqrt(S) if finite variance plausible. Absolute ELBO differs from -KL by unknown log Z; same-target differences are invariant. Never pool draws from different q as though they came from one proposal; a future mixture estimator must use actual mixture density. All four fits may miss a remote mode, yielding reassuring PSIS and ESS despite poor global coverage. Report this limitation.

## 9. Possible multiple basins

Reuse all six J_ab distances, independent ELBO estimates and within-run drift/gradient/R-hat evidence. No extra evaluations. Distinguish evidence compatible with unstable training, similar solutions, and distinct stable solutions; absent calibrated thresholds record inconclusive rather than assign a hard label. Stable disagreement can result from variational optima even for unimodal p, not proof of target modes. Stable agreement cannot exclude other modes. Report all four fits regardless of relative ELBO. Additional starts, bridge/path probes or mixtures are future budgeted interventions, not oracle diagnostics.

## 10. Data model and workflow

Schema vdp/0.1: finite JSON numbers or null with status (never NaN/Infinity). Store large arrays in referenced artifacts with hashes, shape and dtype. Required top-level groups:

- schema_version, protocol_revision, source_commit, opaque target_instance_id, dimension, dtype, environment, transform, configuration, RNG manifest.
- budget: B, charge schedule, used/remaining TEU, N_f/N_g/N_h, stage/run ledgers, attempts/completions/failures/denials, time, memory, stop_reason.
- runs[4]: start/end raw/effective mu/lambda, clamp flags, run status, trajectory reference, costs, gradient check, importance/ELBO diagnostics and curvature diagnostics.
- profile: optimization (R-hat/drift/stationarity/agreement), dependence, tails, geometry, possible_basins. Preserve per-run raw estimates and joint/per-dimension KL.
- evidence: uncertainty methods, assumptions, warnings, missing metrics, recommended investigation or abstention.
- evaluator: separate privileged artifact, inaccessible to inference.

Each metric record: value, status (ok, not_requested, unsupported, insufficient_budget, insufficient_samples, degenerate, nonfinite, run_failed, inconclusive), units, sample_count, estimator_version, uncertainty nullable, warnings. "ok" means computation succeeded, not posterior accuracy. Preserve partial results; no selected-subset four-run diagnostic.

Reference workflow:

    freeze protocol, B, seed IDs, allocations; validate meter
    for each of four runs:
        initialize from declared RNG
        for T steps:
            reserve complete batch before target work
            evaluate metered value_grad on reparameterized draws
            on invalid evaluation record failure; stop this run
            compute explicit gradient, update Adam, log trajectory
        freeze last valid endpoint; flag incomplete runs
    compute trajectory and pairwise diagnostics
    for each eligible endpoint:
        perform fresh gradient check
        perform fresh importance draws / ELBO
        reuse first J locations for R HVP directions
        estimate dependence, geometry and bootstrap sensitivity
    emit VDP, all statuses, costs and recommendation/abstention
    independently audit and evaluate against benchmark truth

No silent restarts or reallocation. On insufficient budget skip unaffordable diagnostic and report why. Endpoint diagnostics may be useful even after partial training if labeled as such.

## 11. Candidate intervention policy (not automated in v0.1)

| Evidence | Investigation | Alternative explanation |
|---|---|---|
| Stable agreement, small drift/gradient, no coverage warnings | Retain MFVI if independent quality evidence supports requirements | Shared wrong solution, missed mode |
| Drift, unstable fluctuations or large gradients | Optimizer, step size, gradient estimator or more budget | Scaling, local optima |
| Stable agreement and off-diagonal curvature | Objective or smallest useful covariance extension | Local curvature only |
| High k or low ESS | Tail/coverage strategy, objective or family | Finite-sample noise, optimization failure |
| Curvature variation/negative directional curvature | Reparameterization or flexible family | Poor fit, local artifact |
| Distinct stable fits | More starts, alternative objective or mixture | Local variational optima |
| Missing/conflicting evidence | Targeted additional information or abstain | Unknown |

Policy must not treat raw diagnostics as calibrated accuracy guarantees. Future experiments compare VDP routing with fixed MFVI, same-budget restart-only, fixed richer methods and oracle routing (evaluator-only). Count all probe/diagnostic/intervention costs. Learn whether VDP predicts intervention benefits, rather than assume it does.


## 12. Verification and staged empirical validation

### Task 002: implementation and Gaussian controls

1. Preserve archived Task 001 runs and protocol; create a versioned Task 002 protocol and fresh output directories. Treat settings in §4.1 as proposed defaults requiring review. Freeze configuration, environment, compute limits, target instances and comparator settings before execution.
2. Test meter: scalar and batch charges, exact boundary, atomic rejection, failed attempts, concurrent reservation if supported, independent instrumentation of target calls, no optimizer mutation on denial, no hidden direct calls. Assert ledger reconciliation and used<=B throughout.
3. Test pathwise gradient against autodiff on analytic quadratic targets with same epsilon; verify clamp derivative, post-update trajectory indexing and independent ELBO. Verify additive log-density shift invariance (same RNG, ELBO shifts by constant); diagonal affine curvature invariance with transformed q; reproducible streams. Verify oracle access is excluded.
4. Test R-hat on synthetic stationary identical, mean-separated, scale-separated, drifting, tied, constant equal/unequal and odd-length traces, missing runs and nonfinite values. Compare nondegenerate cases to pinned reference library. Test drift and pairwise Gaussian KL symmetry/nonnegativity/identity, mean/scale decomposition and scale stability.
5. Test HVP estimators against exact dense oracle on small diagonal, correlated bivariate and general SPD Gaussian matrices. Verify constant curvature variation zero, zero/indefinite H, J=1/R=1 invalid, reproducibility, convergence of energies across repeated random probes and increasing R (not exact equality of noisy ratio). Test varying-curvature toy. Test PSIS and ESS on constant, varying, degenerate and invalid weights, underdispersed Gaussian, log-density shifts and pinned reference fixtures. Test JSON/status and partial-result paths.
6. Development controls: archived independent, correlated and ill-conditioned Gaussian targets at d=10,50,100, four restarts and proposed budget horizons. Preserve Task 001 realized covariances. Add a versioned full-rank seeded normal-matrix QR rotation generator for new instances; record realized matrices and sign convention, avoid historical rank-deficient sine construction. Use opaque instance IDs. Include deliberately short budget and predeclared unstable optimizer controls. Common-start comparisons on a predeclared subset, fully charged. Freeze a bounded experiment matrix before executing it.
7. Independent evaluator reports exact reverse/forward KL, reverse-KL family floor and optimization gap, mean, marginal variance and correlation/covariance errors, joint/per-dimension errors. Compare with diagnostics to test whether incomplete fitting can be distinguished from converged but underdispersed MFVI. Keep all seeds and failed runs. Report seed extrema and variability across target instances separately. No claim that agreement establishes global convergence.
8. Deliver code, tests, frozen protocol, raw results and ledgers, reproducible commands, independent audit, scientific report and limitations on a review branch. No autonomous discovery. Test success precedes empirical claims; unsupported diagnostics are explicitly recorded.

### Task 003: generalization and calibration (separate protocol)

Use banana, funnel, heavy-tailed unimodal targets, balanced and imbalanced mixtures, including distant hidden modes and cases with similar local but different global behavior. Use multiple independently generated target instances. Hold out entire instances/geometries, not merely optimization seeds. Define benchmark truth or reference uncertainty, objective-dependent optimization failure separately from posterior quality, and predeclared quality losses. Freeze development/holdout split, thresholds, uncertainty summaries, and budgets before candidate evaluations. Measure sensitivity/specificity (where labels meaningful), false reassurance, ambiguity/abstention, dimension/coordinate sensitivity, and cost vs horizon/probe count. Use development data only to calibrate gates, then evaluate on untouched holdouts. No numerical threshold is endorsed before validation.

Task 004+ evaluates objective/family/optimizer interventions and predictive usefulness of VDP under matched end-to-end budgets. Compare Pareto quality-cost tradeoffs and worst failures. No composite improvement can compensate for critical reliability failure. Return unresolved warnings when no validated gate is attainable within B. Finite black-box evaluation cannot certify global posterior coverage for arbitrary densities.

## 13. Literature anchors and novelty boundary

- Dhaka et al. (2020), [Robust, Accurate Stochastic Optimization for Variational Inference](https://proceedings.neurips.cc/paper/2020/hash/7cac11e2f46ed46c339ec3d569853759-Abstract.html), NeurIPS. Existing work treats stochastic optimization as a Markov process and develops convergence diagnostics and stopping rules. Applying R-hat to VI must not be claimed as novel without careful comparison.
- Dhaka et al. (2021), [Challenges and Opportunities in High Dimensional Variational Inference](https://proceedings.neurips.cc/paper_files/paper/2021/hash/404dcc91b2aeaa7caa47487d1483e48a-Abstract.html), NeurIPS. Objective/family/optimization tradeoffs motivate our research.
- Vehtari et al. (2021), [Rank-normalization, folding, and localization: An improved R-hat for assessing convergence of MCMC](https://doi.org/10.1214/20-BA1221), Bayesian Analysis. MCMC reference statistic, requiring separate validation for VI trajectories.
- Yao et al. (2018), [Yes, but Did It Work?: Evaluating Variational Inference](https://proceedings.mlr.press/v80/yao18a.html), ICML. Existing PSIS and simulation-based calibration for VI assessment.
- Vehtari et al. (2024), [Pareto Smoothed Importance Sampling](https://jmlr.org/papers/v25/19-556.html), JMLR. Modern PSIS and sample-size-dependent diagnostic references.

The candidate contribution is the budgeted joint difficulty profile, its empirical predictive value for intervention selection, and the resulting adaptive inference policy. Individual diagnostic techniques are not automatically novel. A broader prior-art review is required before a novelty claim.

## 14. Decisions requiring review before execution

- Four runs and unnormalized target access are project requirements; TEU weights, sample counts, seed derivation, initialization spread, diagnostic windows, and fixed training horizons are proposed engineering conventions.
- Choose and pin the exact PSIS/R-hat reference library and dependency versions in implementation; confirm reference output conventions. Do not silently reinterpret metrics.
- Freeze the explicit benchmark matrix, unstable-optimizer comparator, common-start subset, and quality losses/gates as appropriate to each stage. Record the cost ceiling for the entire experiment separately from each inference run.
- This is a specification, not implemented software or validated experimental evidence. Keep Task 001 unchanged and update the review branch only.
