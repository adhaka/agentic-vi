"""Run one or more reproducible MFVI Gaussian baseline experiments."""
import argparse
import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from agentic_vi.algorithms.mfvi import MFVIConfig, MeanFieldGaussianVI
from agentic_vi.targets import make_gaussian_target
from agentic_vi.evaluation import evaluate_mfvi


def run_one(args, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    target = make_gaussian_target(args.dimension, args.target, dtype=dtype)
    config = MFVIConfig(
        dimension=args.dimension,
        steps=args.steps,
        learning_rate=args.learning_rate,
        mc_samples=args.mc_samples,
        log_every=args.log_every,
        dtype=dtype,
    )
    model = MeanFieldGaussianVI(config)
    fit_result = model.fit(target)
    metrics = evaluate_mfvi(model, target)
    mean, variance = model.moments()

    return {
        "evaluation_schema": 2,
        "experiment": "mfvi_gaussian_baseline_v0",
        "seed": seed,
        "target": args.target,
        "dimension": args.dimension,
        "config": {
            "log_every": args.log_every,
            "steps": args.steps,
            "learning_rate": args.learning_rate,
            "mc_samples": args.mc_samples,
            "dtype": args.dtype,
        },
        "covariance_sha256": hashlib.sha256(target.covariance.numpy().tobytes()).hexdigest(),
        "fitted_mean": mean.tolist(),
        "fitted_variance": variance.tolist(),
        "runtime_seconds": fit_result["runtime_seconds"],
        "metrics": metrics,
        "training_history": fit_result["history"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dimension", type=int, default=10)
    parser.add_argument("--target", choices=["independent", "correlated", "ill_conditioned"],
                        default="correlated")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--mc-samples", type=int, default=128)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else [args.seed]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        result = run_one(args, seed)
        output = args.output_dir / f"{args.target}_d{args.dimension}_seed{seed}.json"
        output.write_text(json.dumps(result, indent=2))
        print(f"Saved {output}")
        print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
