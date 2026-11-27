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
