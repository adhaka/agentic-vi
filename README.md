# Agentic VI — starter research repository

**Research scope:** autonomous discovery of complete VI algorithms  
**Initial implementation:** mean-field Gaussian VI (MFVI), reverse-KL / ELBO training  
**Language:** Python + PyTorch

## 1. What this first milestone does

This repository establishes a controlled baseline on synthetic multivariate Gaussian targets, for which the exact posterior is known.

It:
- optimizes a diagonal Gaussian approximation using the reparameterization trick;
- maximizes the ELBO using Monte Carlo samples;
- evaluates the result using exact `KL(q || p)` and moment errors;
- records the seed, configuration, runtime, and metrics in JSON.

For a Gaussian target `p = N(m, Σ)`, the reverse-KL-optimal diagonal Gaussian has
`mu_i = m_i` and `var_i = 1 / (Σ^{-1})_ii`. These are generally **not** the
marginal variances `Σ_ii`. This makes correlated Gaussian targets a useful
first test of mean-field limitations.

## 2. Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e .
python run_baseline.py --dimension 10 --target correlated --seed 0
```

Run a small seed sweep:

```bash
python run_baseline.py --dimension 10 --target correlated --seeds 0 1 2 3 4
```

## 3. Repository boundaries

- `agentic_vi/algorithms/mfvi.py`: the initial algorithm implementation.
- `agentic_vi/targets.py`: synthetic target definitions.
- `agentic_vi/evaluation.py`: independent metrics and exact Gaussian KL.
- `run_baseline.py`: experiment entry point.
- `protocol.yaml`: initial fixed experiment protocol.

In the eventual autonomous loop, the agent may propose changes to algorithm code.
The target definitions, evaluator, protocol, and held-out benchmark suite should be
read-only to the agent. Keep version-control commits for every accepted change.

## 4. Initial success criteria

Do not declare MFVI successful merely because its ELBO increases. For each run,
inspect:
- finite parameters and metrics;
- convergence behavior;
- exact `KL(q || p)` on the Gaussian test target;
- mean and variance errors;
- variability across seeds.

The exact KL is an evaluation metric available only because this initial target is
synthetic and normalized. For general Bayesian models, the normalizing constant
is unknown; the project will need a portfolio of diagnostics rather than exact KL.

## 5. Important scope limitation

This is a **baseline scaffold**, not yet an autonomous discovery agent and not
evidence of a new VI method. It intentionally starts with one target family,
one variational family, one objective, and one optimizer. The next milestone is
validation and benchmark expansion, not unrestricted automated search.
