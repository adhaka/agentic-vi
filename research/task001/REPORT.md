# Task 001 · Gaussian mean-field baseline

## Question and design

How much reverse KL comes from imperfect stochastic fitting, and how much remains at the best diagonal Gaussian? We crossed three fixed Gaussian geometries with dimensions 10, 50 and 100. Each cell used training seeds 0–4. Within a cell, seeds changed the optimization draws and shared one target. Means range linearly from −0.5 to 0.5. Independent variances range from 0.5 to 2. The correlated targets use scaled Toeplitz correlations 0.75^|i−j|. The rotated ill-conditioned targets have eigenvalues from 1 to 0.001.

Starting at zero means and unit variances, Adam took 3,000 steps at learning rate 0.03 using 128 reparameterized samples per step in float64. Each fit used 384,000 training draws. The 45 fits used 135,000 updates and 17,280,000 draws in total. Exact evaluation added zero random draws. The log-standard-deviation clamp [−12, 8] contains the analytical optimum for every target here. This is a matched update and sample budget. Dense quadratic forms cost more as dimension increases.

## Fitting and approximation errors

KL is in nats per joint distribution. The floor is common to all seeds in a cell. Gap is excess KL above the floor. The parenthesized seed attained the largest gap. Fit time is mean wall time per training loop. Floor values within 10⁻¹² of zero are displayed as zero.

| Target | d | Floor | Mean gap | Gap range (worst seed) | Mean total KL | Fit s |
|---|---:|---:|---:|---:|---:|---:|
| independent | 10 | 0 | 0.01063 | 0.006462–0.01889 (3) | 0.01063 | 2.071 |
| independent | 50 | 0 | 0.07706 | 0.07019–0.08504 (2) | 0.07706 | 2.961 |
| independent | 100 | 0 | 0.1582 | 0.137–0.1751 (3) | 0.1582 | 3.881 |
| correlated | 10 | 2.198 | 0.01893 | 0.007327–0.0386 (3) | 2.217 | 2.044 |
| correlated | 50 | 11.12 | 0.1138 | 0.08749–0.1498 (2) | 11.24 | 2.778 |
| correlated | 100 | 22.28 | 0.2379 | 0.1802–0.2718 (0) | 22.52 | 3.754 |
| ill_conditioned | 10 | 7.679 | 0.03288 | 0.009102–0.04485 (1) | 7.712 | 2.059 |
| ill_conditioned | 50 | 35.71 | 0.3261 | 0.1105–0.5183 (0) | 36.03 | 2.891 |
| ill_conditioned | 100 | 72.31 | 0.6585 | 0.4209–0.8717 (4) | 72.97 | 3.679 |

## Moment recovery

Each MSE first averages squared coordinate errors, then averages over seeds. Variance MSE is scale-dependent. The last column gives the extreme fitted-to-posterior variance ratios across every coordinate and seed in the cell. Ratios below one indicate underdispersion. Per-seed diagnostics and full fitted moments accompany the report.

| Target | d | Mean MSE | Variance MSE vs MF optimum | Variance MSE vs posterior | Marginal variance ratio range |
|---|---:|---:|---:|---:|---:|
| independent | 10 | 0.00147 | 0.004032 | 0.004032 | 0.8695–1.078 |
| independent | 50 | 0.001361 | 0.006724 | 0.006724 | 0.8191–1.183 |
| independent | 100 | 0.001515 | 0.006134 | 0.006134 | 0.7953–1.204 |
| correlated | 10 | 0.0005819 | 0.000443 | 0.6498 | 0.2561–0.4608 |
| correlated | 50 | 0.0005087 | 0.000467 | 0.6921 | 0.2304–0.473 |
| correlated | 100 | 0.000575 | 0.0004178 | 0.694 | 0.2272–0.4564 |
| ill_conditioned | 10 | 1.582e-05 | 1.787e-07 | 0.03391 | 0.0132–0.1084 |
| ill_conditioned | 50 | 3.431e-05 | 1.229e-05 | 0.02144 | 0.02521–0.43 |
| ill_conditioned | 100 | 3.769e-05 | 1.513e-05 | 0.01798 | 0.03121–0.7535 |

## Interpretation and limits

All 45 fits finished with finite parameters and metrics. The archived reports passed the independent numerical audit. On independent targets the exact family floor is zero. Mean joint optimization gap grew from 0.0106 at dimension 10 to 0.158 at dimension 100, while mean-coordinate MSE stayed near 0.0015. Joint KL accumulates coordinate errors, so its growth alone cannot establish optimizer instability.

On the correlated target at dimension 100, the family floor explains 22.28 of 22.52 mean KL nats (98.9%). Fitted variances ranged from 23% to 46% of their target marginals even though MSE relative to the optimal diagonal variance was only 0.000418. An ideal optimizer restricted to the same family could recover only the 0.238-nat average gap.

The rotated target at dimension 100 had both the largest mean optimization gap (0.659 nats) and the largest family floor (72.31 nats). The floor accounts for 99.1% of mean KL. Seed 4 incurred a gap of 0.872 nats. Across rotated targets, the smallest coordinate variance ratio was 0.0132 at dimension 10. Absolute variance MSE alone makes this severe relative underdispersion easy to miss because the targets differ in scale.

These observations support testing covariance structure if the goal is calibrated marginal uncertainty on these targets. A change in step size could affect the residual fitting gap, but this study did not compare optimizers, schedules, or richer families. The five seeds measure optimizer variability conditional on one target per cell. They give no estimate of variability across rotations or datasets. All targets are Gaussian, axis alignment is a property of the chosen parameterization, and results cover dimensions up to 100. There is no held-out reliability or speed comparison.

## Computational record

The saved environment restored an importable editable checkout and its Python 3.12.14, PyTorch 2.9.1+cpu and NumPy 2.2.6 environment without a reinstall. The matrix ran sequentially on an AMD EPYC 9V74 with one intra-op and one inter-op thread. The driver took 131.4 s elapsed and 131.3 s process CPU time after imports. Peak process RSS was 241.8 MiB. Training-loop times summed to 130.6 s. Individual JSONs record the loop time, configuration, seed, covariance hash, final moments, diagnostics, and 31 training observations. The final observation is a pre-update stochastic estimate. The reported final ELBO is the exact post-update value.

## Correctness review and validation

- The implemented pathwise objective is a Monte Carlo expectation of normalized log p minus log q. The scale parameter is a standard deviation and the evaluator correctly squares it to get variance. The original total KL formula agrees with the closed form.
- The original runner labeled the final training minibatch estimate as the fitted ELBO. It also used an ambiguous name for variance error relative to the family optimum. Evaluation schema 2 corrects the ELBO reference and reports both variance comparisons. See [metric compatibility](../../DECISIONS.md).
- Four independent-oracle unit tests cover a bivariate conditional-variance solution and perturbed fit, an independent exact solution, target densities against PyTorch's Cholesky-based Gaussian for the nine cells, and the runner's post-update ELBO after one optimizer step. The experiment audit independently integrates the quadratic log-density ratio with a symmetric 2d-point rule in NumPy, using archived moments and covariances. It checks every seed, configured budget, history, and canonical metric.
- Targets use an explicit inverse and a log absolute determinant. These operations require a valid positive-definite input for the density interpretation. All nine archived matrices admit Cholesky factors. Ill-conditioned QR starts from entries sin(i·d + j + 1) for zero-based indices. The sine addition formula bounds that matrix's exact rank by two. QR still returns an orthogonal basis, but its completion depends on numerical details. The realized matrices are included for cross-platform verification.

## Reproduction and evidence

Run from the repository root with the restored environment. The experiment command requires a new output directory to preserve prior results.

```bash
source /workspace/.venvs/agentic-vi/bin/activate
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -v
python research/task001/audit.py research/task001/results
python research/task001/run.py /tmp/agentic-vi-task001-reproduction
python research/task001/audit.py /tmp/agentic-vi-task001-reproduction
```

[Per-seed metrics](per_seed.csv) · [Raw fits and covariance matrices](results/) · [Source hashes, versions, costs, and failure ledger](results/manifest.json) · [Analytical derivation](../../RESEARCH.md)
