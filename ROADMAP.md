# Research milestones

1. **Gaussian baseline (Task 001).** Establish an independent analytical oracle, freeze the training protocol, and measure all nine target/dimension cells over five seeds. See research/task001/REPORT.md for results and limitations.
2. **Resolve optimization failures.** If the gap dominates, inspect precision-weighted mean and variance contributions. Test a single budget-matched intervention on development targets and retain all seed results.
3. **Address approximation failures.** Where an irreducible floor and marginal underdispersion persist despite near-optimal fitting, compare the smallest covariance extension justified by the geometry. Include its parameter, sample, and time costs.
4. **Broaden validation.** Design held-out target instances, include non-Gaussian geometries, choose calibrated tail and uncertainty diagnostics, and specify failure criteria before evaluating candidates.
5. **Consider autonomous experiments.** Require protected evaluation, budget accounting, held-out checks, reproducible proposals, and human review before building a discovery loop.
