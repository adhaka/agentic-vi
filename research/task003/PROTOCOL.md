# Task 003 protocol, revision 1

Compare centered (C) and non-centered (NC) representations of a single Eight Schools posterior. The question is whether VDP separates their geometry, optimization and coverage at matched budgets, and whether the differences track scientific accuracy. Direction of improvement is open. Use Task 002 at 977e020. Preserve earlier data and code.

## Target and initialization

Use the data and normalized priors in the task request, independent observations yᵢ given θᵢ with known standard errors, and independent Normal(μ,τ²) school effects conditional on μ,τ. Sample in (μ,η,w₁,…,w₈), η=log τ. In C, w=θ. In NC, w=z with θ=μ+exp(η)z. Include η for the scale transform. The additional NC determinant exp(8η) cancels the centered conditional scale factors. Retain all model normalization constants for matching the quadrature log evidence.

Use the existing zero/unit reference, dispersed initialization and four root IDs. Identical numerical initial means and log scales refer to θ directly in C, to standardized deviations in NC. In NC all scientific effects share μ and τ even in an initially diagonal proposal. Thus both the initial scientific distributions and the attainable scientific families differ. This design measures the whole parameterization choice for diagonal MFVI.

## Matrix and checks

Two representations × T ∈ {250,500,1000,3000}, four runs per cell. CPU float64, reverse KL, diagonal Gaussian, constant Adam lr 0.03, 64 samples/update, existing clamp and stream derivation. Use G=256, S=1024, J=8, R=16 and 200 bootstrap resamples. All diagnostics are Task 002 defaults. Initialization and training draws are paired across horizons and representations. Diagnostic draws are fresh across horizons and paired across representations for the same root and horizon. Use opaque instance IDs for inference. The budgets per profile are 135,680, 263,680, 519,680 and 1,543,680 TEU. Total ceiling 4,925,440 TEU. No smoke profiles planned: derivative, equivalence and reference tests plus the existing VDP suite precede execution. Retain failures with ledgers; do not replace runs.

## Independent evaluation

Integrate the schools and μ analytically given τ. Integrate the remaining one-dimensional posterior using quadrature on the half-Cauchy prior probability scale. Compute log evidence, marginal means, SDs and 5th/50th/95th percentiles for μ, τ and each θ. Verify quadrature by refinement and a separate adaptive integral using the multivariate marginal likelihood. For each fitted diagonal Gaussian compute reverse KL as log evidence minus expected log joint plus expected log q. Evaluate Gaussian/lognormal moments analytically; use scalar quadrature where needed for log-prior expectations and transformed school quantiles. Keep evaluator inputs out of fit and profile construction. Report signed coordinate errors, including SD, and summarize school errors using mean absolute error across eight schools. Compare raw diagnostics, seed ranges, agreement, both signs of curvature where sampled and resource use. No threshold-based choice of parameterization. This is a paired descriptive study with one data set and four training streams.
