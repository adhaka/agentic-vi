# Research milestones

1. **Gaussian baseline (Task 001).** Establish an independent analytical oracle, freeze the training protocol, and measure all nine target/dimension cells over five seeds. See research/task001/REPORT.md for results and limitations.
2. **Resolve optimization failures.** If the gap dominates, inspect precision-weighted mean and variance contributions. Test a single budget-matched intervention on development targets and retain all seed results.
3. **Address approximation failures.** Where an irreducible floor and marginal underdispersion persist despite near-optimal fitting, compare the smallest covariance extension justified by the geometry. Include its parameter, sample, and time costs.
4. **Broaden validation.** Design held-out target instances, include non-Gaussian geometries, choose calibrated tail and uncertainty diagnostics, and specify failure criteria before evaluating candidates.
5. **Consider autonomous experiments.** Require protected evaluation, budget accounting, held-out checks, reproducible proposals, and human review before building a discovery loop.


## Proposed diagnostic-driven sequence (prospective revision)

After Task 001, prioritize a four-run MFVI pilot diagnostic study (Task 002 protocol design) alongside the analytical objective/family reference comparisons. Establish trajectory logging, R-hat edge-case handling and distribution agreement metrics; validate on Gaussian unique-optimum controls and deliberate optimization failures. Expand to non-Gaussian unimodal and multimodal instances. Then evaluate diagnostic-directed objective, family and optimizer interventions under a matched end-to-end budget against fixed-method baselines. Freeze held-out targets and quality gates before testing. Autonomous discovery remains gated on review of this evidence.
