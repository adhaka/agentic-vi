# Research charter

Agentic VI studies how simple, computationally efficient variational methods can yield reliable posterior approximations as dimension and problem geometry change. The motivation comes from Karpathy's Autoresearch and Dhaka et al.'s work on reliable and high-dimensional VI (NeurIPS 2020 and 2021). The current evidence concerns normalized Gaussian targets with a known solution.

## Gaussian reference problem

Let the target be a nonsingular Gaussian with mean m, covariance Σ and precision P. Let q be Gaussian with mean μ and diagonal covariance D with positive entries vᵢ. Substituting the normalized Gaussian log densities and integrating under q gives

    2 KL(q || p) = tr(PD) + (μ − m)ᵀP(μ − m) − d + log det Σ − Σᵢ log vᵢ.

The gradient in μ is P(μ − m). The derivative in each vᵢ is (Pᵢᵢ − 1/vᵢ)/2. The Hessian has positive-definite mean block P and positive variance entries 1/(2vᵢ²), with zero cross derivatives. Thus the unique optimum has μ* = m and vᵢ* = 1/Pᵢᵢ. Each vᵢ* equals the conditional variance given the remaining coordinates. The Schur complement gives vᵢ* ≤ Σᵢᵢ, with equality for a coordinate independent of the others.

Substitution splits the total reverse KL into two nonnegative terms:

    floor = ½ (log det Σ + Σᵢ log Pᵢᵢ)
    gap   = ½ [(μ − m)ᵀP(μ − m) + Σᵢ (rᵢ − 1 − log rᵢ)],   rᵢ = Pᵢᵢ vᵢ.

Hadamard's inequality applied to P gives floor ≥ 0, with equality for diagonal precision. Each variance term in the gap is nonnegative by log r ≤ r − 1. For a normalized target the exact ELBO is −KL. The gap therefore measures ELBO lost by fitting within this family. The floor measures ELBO lost by using this family even with perfect optimization. The final exact ELBO and total KL express the same diagnostic.

## Reliability criteria

Report the fitting gap and family floor alongside posterior mean error and two distinct variance errors. A small gap can coexist with severely underestimated marginal uncertainty. Report computational cost and seed variability separately. No composite score or universal pass threshold has been established. Training traces contain noisy pre-update ELBO estimates. They cannot certify stationarity. These oracle comparisons require known target structure and do not directly transfer to unknown posteriors.


## Expanded research agenda: objective, family, optimizer and adaptation

Our hypothesis is that diagnostic-driven selection of the objective, variational family and optimizer can improve reliability at controlled computational cost. Reliability before efficiency; simplicity before complexity; evidence before acceptance. This is a hypothesis, not a novelty claim.

Represent an algorithm by (Q, D, O, S): family, estimable training objective, optimizer and adaptive strategy. Seek low cost subject to predeclared complementary quality constraints on development and held-out targets. Report Pareto trade-offs and worst-case failures; no aggregate improvement can compensate for violating a critical constraint. Thresholds must be justified and frozen before comparisons.

Task 001's floor is specific to reverse KL AND the diagonal family, not an objective-independent expressivity bound. For Gaussian p, forward KL within the same family instead has mu=m and v_i=Sigma_ii (minimize the expected negative log q under p). This recovers marginal moments but not correlations. Oracle forward KL is an analytical comparator only: target samples and exact moments are forbidden in deployable black-box training. Exact forward KL can be computed as half [sum_i Sigma_ii/v_i + sum_i (mu_i-m_i)^2/v_i - d + sum_i log v_i - log det Sigma].

### Proposed Task 002 (protocol design before experiments)

Cross objective choice and covariance structure rather than conflating them. Begin with analytical forward/reverse KL diagonal optima, then compare the existing stochastic reverse-KL baseline against a full-rank or justified structured Gaussian control. Investigate practical mass-covering estimators (e.g. self-normalized importance-weighted forward-KL gradients) only with explicit bias, weight degeneracy, cost and numerical stability accounting. Do not use oracle information in those estimators. Extend to Renyi/alpha objectives with explicit conventions and estimators. The 2021 Dhaka et al. paper (https://proceedings.neurips.cc/paper_files/paper/2021/hash/404dcc91b2aeaa7caa47487d1483e48a-Abstract.html) provides a motivation: mass-covering objectives may be difficult to optimize in high dimensions. Test rather than assume that conclusion generalizes.

Use independent evaluation of reverse and forward KL on known Gaussian targets, marginal variance ratios, mean error, correlation/covariance error, and computational cost. Exact comparisons remain evaluation-only. For unknown-normalizer targets, use complementary ELBO, importance diagnostics (PSIS k and ESS), Stein discrepancy where suitable, predictive performance and repeated simulation-based calibration; none alone certifies accuracy. Track unavailable or invalid metrics explicitly. Distinguish optimization gaps within each objective from cross-objective posterior quality. Do not interpret differences between unrelated objective values as additive errors.

Before search, version the benchmark to include full-rank seeded random rotations and multiple target instances, preserve Task 001 archived matrices, and reserve held-out target instances. Compare objective/family interventions with matched target density/gradient evaluation budgets and report actual time, memory, parameter count and failures. Define the target access contract (unnormalized log density and derivatives only for deployable algorithms). No autonomous loop until these checks and protocol are reviewed.
