# Research decisions

## Task 001

- Use the initial implementation at 401dc9d as the training reference. Extend the evaluation before the experiment. Preserve protocol.yaml and the target and algorithm sources at their original revisions.
- Cross the requested three dimensions and three target kinds with the protocol's five seeds. Use the same settings in every run. Fix the CPU thread count and run sequentially. Equal updates and draws permit sample-budget comparison. Larger matrices still cost more per update.
- Report the normalized-target ELBO analytically after the last optimizer step. Keep the stochastic trace for training diagnostics. Schema 2 is recorded in each new JSON. Historical v0 records keep their original interpretation.
- Use posterior_mean_mse, posterior_marginal_variance_mse, and optimal_mean_field_variance_mse as the canonical names. mean_squared_error and diagonal_variance_mse are compatibility aliases for the first and third. The latter legacy key has never measured posterior marginal variance error.
- Report all seeds without tuning or filtering. Use per-cell means for scale, seed extrema for reliability, and raw coordinate moments for follow-up. Thresholds for a downstream decision require a specified loss or coverage target.
- Preserve the ill-conditioned target construction for comparability. Archive each realized covariance because its QR completion can vary across numerical libraries. Any replacement rotation belongs to a versioned future benchmark.
