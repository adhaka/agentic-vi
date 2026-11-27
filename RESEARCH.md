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
