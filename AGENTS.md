# Working on Agentic VI

Read RESEARCH.md, ROADMAP.md, DECISIONS.md, and the latest reviewed experiment report before proposing a change. Use the existing checkout. Prepare research changes on a review branch. Merge only after review.

## Experiment boundaries

- Freeze the target definitions, evaluator, protocol, seeds, and compute budget before testing a candidate. A change to any of these requires a separately described and reviewed benchmark revision. Preserve prior results and schema meanings.
- Keep training independent of oracle metrics. Optimizers may use the target log density and gradients. Exact posterior moments, KL and optima are evaluation data for this benchmark.
- Record source revision, exact settings, software versions, target identity, seed, output, elapsed time, and failures. Keep per-seed results. Use a fresh output directory for each study.
- Compare algorithms at the same sample and update budgets. Report wall time separately. Inspect individual diagnostics and worst runs before proposing an improvement.
- Keep hypotheses distinct from observations. Start with the smallest intervention that addresses a measured failure. Autonomous discovery requires a separate milestone.

## Validation

From the repository root, activate the environment and run the analytical tests:

```bash
source /workspace/.venvs/agentic-vi/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

Task 001 includes the experiment runner, audit command, raw data, and interpretation under research/task001.


## Objective and family discovery rules (proposed Task 002)

- Treat objective, family, optimizer and adaptation as distinct experimental factors. Change one factor at a time when diagnosing effects; use factorial comparisons when testing interactions.
- Every training objective must specify direction, normalization convention, estimator, bias/variance limitations, target access requirements and cost. Oracle target samples/moments are evaluation-only; label oracle comparators explicitly.
- Never select candidates solely by their own training objective. Freeze complementary quality gates before running candidates, report all failures, and prefer simpler feasible methods. Do not silently tune thresholds or reuse held-out targets for development.
- Count target log-density and gradient calls including adaptation and importance weighting; report wall time and memory separately. Matched steps alone are not matched compute.
- Task 002 is a proposed controlled study, not permission to start autonomous search. Preserve Task 001 and prepare a versioned protocol for review before execution. Repository rules alone are not a security boundary: future agent execution must enforce evaluator and benchmark isolation.


## MFVI diagnostic pilot rules

- For prospective studies start with four independent MFVI runs using frozen seed identifiers and a documented dispersed initialization policy; log separate initialization and sampling RNG streams. Use common-start controls when isolating stochastic-gradient variability. Do not rewrite Task 001's five-seed records.
- Record parameter trajectories at fixed cadence; freeze the diagnostic window and compute classical and rank-normalized split/folded R-hat, late-window drift, gradient uncertainty and all six fitted-distribution pairwise comparisons. Explicitly handle zero variance, insufficient observations and nonfinite values. Never interpret R-hat as posterior MCMC convergence or a proof of global optimality.
- Maintain a diagnostic difficulty profile (optimization, dependence, tails/coverage, geometry, possible multiple basins); record evidence, uncertainty and the rationale for the next intervention. Agreement does not rule out missed modes. Disagreement does not establish multimodality.
- Keep oracle labels and metrics out of deployable decisions. Charge all four runs, diagnostics and follow-up work to the same end-to-end budget. Stop only at predeclared reliability gates, otherwise investigate or report unresolved failure. Validate the policy on controlled and held-out targets before authorizing autonomous discovery.
