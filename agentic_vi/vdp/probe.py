"""Four-run diagonal Gaussian probe. All target calls use fixed scoped allocations."""

from dataclasses import asdict, dataclass
from itertools import combinations
import platform
import resource
import time

import numpy as np

from .budget import Denied
from .schema import finalize
from .diagnostics import agreement, curvature, drift, gradient, importance, metric, rhat


@dataclass(frozen=True)
class Config:
    steps: int
    batch: int = 64
    G: int = 256
    S: int = 1024
    J: int = 8
    R: int = 16
    bootstrap: int = 200
    lr: float = 0.03
    common_start: bool = False

    def __post_init__(self):
        if (
            min(self.steps, self.batch, self.bootstrap) < 1
            or min(self.G, self.J, self.R) < 2
            or self.S < self.J
        ):
            raise ValueError("invalid sampling design")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("invalid learning rate")

    def full_cost(self):
        return 4 * (
            2 * self.steps * self.batch + 2 * self.G + self.S + 3 * self.J * self.R
        )

    def allocations(self, budget):
        quotas = {}
        for run in range(4):
            remaining = budget // 4
            for stage, need in [
                ("gradient", 2 * self.G),
                ("importance", self.S),
                ("curvature", 3 * self.J * self.R),
                ("train", 2 * self.batch * self.steps),
            ]:
                quotas[f"{run}/{stage}"] = min(need, remaining)
                remaining -= quotas[f"{run}/{stage}"]
        return quotas


def streams(run, horizon):
    tags = ["init", "train", "gradient", "importance", "curvature", "bootstrap"]
    seeds = {
        name: int(
            np.random.SeedSequence(
                [1, run, tag, 0 if tag < 2 else horizon]
            ).generate_state(1, dtype=np.uint64)[0]
        )
        for tag, name in enumerate(tags)
    }
    return seeds, {name: np.random.default_rng(seed) for name, seed in seeds.items()}


def pathwise(g, eps, raw):
    scale = np.exp(np.clip(raw, -12, 8))
    return np.stack(
        [
            g.mean(axis=0),
            ((raw >= -12) & (raw <= 8)) * (1 + (g * eps * scale).mean(axis=0)),
        ]
    )


def endpoint(p):
    return dict(
        mu=p[0].copy(),
        raw_lambda=p[1].copy(),
        effective_lambda=np.clip(p[1], -12, 8),
        clamp_flags=(p[1] <= -12) | (p[1] >= 8),
    )


def fit(api, p, config, rng):
    first = np.zeros_like(p)
    second = np.zeros_like(p)
    states = [p.copy()]
    gradients, objectives = [], []
    status, reason = "ok", None
    for step in range(1, config.steps + 1):
        lam = p[1].clip(-12, 8)
        eps = rng.standard_normal((config.batch, p.shape[1]))
        try:
            values, g = api.value_grad(p[0] + np.exp(lam) * eps)
            grad = pathwise(g, eps, p[1])
            first = 0.9 * first + 0.1 * grad
            second = 0.999 * second + 0.001 * grad**2
            candidate = p + config.lr * (first / (1 - 0.9**step)) / (
                np.sqrt(second / (1 - 0.999**step)) + 1e-8
            )
            obj = values.mean() + lam.sum() + len(lam) * (1 + np.log(2 * np.pi)) / 2
            if (
                not np.isfinite(candidate).all()
                or not np.isfinite(obj)
                or not np.isfinite(second).all()
            ):
                raise FloatingPointError("nonfinite optimizer update")
        except Exception as exc:
            status = "insufficient_budget" if isinstance(exc, Denied) else "run_failed"
            reason = str(exc)
            break
        p = candidate
        states.append(p.copy())
        gradients.append(grad)
        objectives.append(obj)
    traj = dict(
        raw_states=np.asarray(states),
        preupdate_gradient=np.asarray(gradients).reshape(-1, *p.shape),
        preupdate_objective=np.asarray(objectives),
        lr=config.lr,
        teu_per_step=2 * config.batch,
    )
    return p, traj, status, reason


def diagnostic(fn, n, units):
    try:
        return fn()
    except Exception as exc:
        status = (
            "insufficient_budget"
            if isinstance(exc, Denied)
            else "unsupported"
            if isinstance(exc, NotImplementedError)
            else "run_failed"
        )
        if isinstance(exc, FloatingPointError):
            status = "nonfinite"
        return metric(
            None,
            units,
            n if status in ("nonfinite", "run_failed") else 0,
            status,
            warnings=[str(exc)],
        )


def profile(
    d, config, scope, budget_report, *, target_id, source, hvp=True, reference=None
):
    """scope(run/stage) yields Primitives. Oracle quantities never enter this interface."""
    start = time.perf_counter()
    runs, windows, trajectories, manifest = [], [], [], []
    base, scale = (
        (np.zeros(d), np.ones(d)) if reference is None else map(np.asarray, reference)
    )
    if (
        base.shape != (d,)
        or scale.shape != (d,)
        or not np.isfinite([base, scale]).all()
        or np.any(scale <= 0)
    ):
        raise ValueError("invalid reference")
    for i in range(4):
        tic = time.perf_counter()
        seeds, rng = streams(i, config.steps)
        manifest.append(seeds)
        p = np.stack([base, np.log(scale)])
        if not config.common_start:
            p[0] += scale * rng["init"].standard_normal(d)
            p[1] += 0.25 * rng["init"].standard_normal(d)
        init = endpoint(p)
        p, traj, status, reason = fit(scope(f"{i}/train"), p, config, rng["train"])
        trajectories.append(traj)
        end = endpoint(p)
        raw = traj["raw_states"][1:]
        n = len(raw) // 2
        win = np.concatenate([raw[:, 0], raw[:, 1].clip(-12, 8)], axis=1)[
            len(raw) - n :
        ]
        windows.append(win)
        rec = dict(
            id=i,
            start=init,
            end=end,
            steps=len(raw),
            status=status,
            reason=reason,
            drift=drift(win),
        )
        runs.append(rec)
        if status == "run_failed":
            for key in ("gradient", "importance", "curvature"):
                rec[key] = metric(None, "unavailable", 0, "run_failed")
        else:
            mu, sigma = p[0], np.exp(end["effective_lambda"])
            eps = rng["gradient"].standard_normal((config.G, d))
            rec["gradient"] = diagnostic(
                lambda: gradient(
                    scope(f"{i}/gradient").value_grad(mu + eps * sigma)[1], eps, sigma
                ),
                config.G,
                "standardized",
            )
            if end["clamp_flags"].any():
                rec["gradient"]["warnings"].append(
                    "clamped endpoint, effective-parameter gradient"
                )
                if rec["gradient"]["status"] == "ok":
                    rec["gradient"]["status"] = "inconclusive"
            eps = rng["importance"].standard_normal((config.S, d))
            x = mu + eps * sigma
            logq = (
                -0.5 * (eps**2 + np.log(2 * np.pi)).sum(axis=1)
                - end["effective_lambda"].sum()
            )
            rec["importance"] = diagnostic(
                lambda: importance(scope(f"{i}/importance").value(x) - logq),
                config.S,
                "weights and nats",
            )

            def measure_curvature():
                z = rng["curvature"].choice([-1.0, 1.0], size=(config.R, d))
                points = np.repeat(x[: config.J], config.R, axis=0)
                directions = np.tile(z * sigma, (config.J, 1))
                _, _, hz = scope(f"{i}/curvature").value_grad_hvp(points, directions)
                return curvature(
                    (hz * sigma).reshape(config.J, config.R, d),
                    z,
                    rng["bootstrap"],
                    config.bootstrap,
                )

            rec["curvature"] = (
                diagnostic(
                    measure_curvature, config.J * config.R, "standardized curvature"
                )
                if hvp
                else metric(None, "standardized curvature", 0, "unsupported")
            )
        rec["seconds"] = time.perf_counter() - tic
    valid = all(r["status"] != "run_failed" for r in runs)
    rh = rhat(np.stack(windows) if valid and len(set(map(len, windows))) == 1 else None)
    matrix = [
        [
            0.0 if a == b and runs[a]["status"] != "run_failed" else None
            for b in range(4)
        ]
        for a in range(4)
    ]
    pairs = {}
    for i, j in combinations(range(4), 2):
        if "run_failed" in (runs[i]["status"], runs[j]["status"]):
            pairs[f"{i}-{j}"] = metric(None, "nats", 0, "run_failed")
            continue
        a, b = (runs[k]["end"] for k in (i, j))
        v = agreement(
            (a["mu"], a["effective_lambda"]), (b["mu"], b["effective_lambda"])
        )
        pairs[f"{i}-{j}"] = metric(v, "nats", 2)
        matrix[i][j] = matrix[j][i] = v["joint"]
    vals = [p["value"]["joint"] for p in pairs.values() if p["status"] == "ok"]
    budget = budget_report()
    budget.update(
        seconds=time.perf_counter() - start,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        stop_reason="fixed_horizon"
        if all(r["status"] == "ok" for r in runs)
        else "incomplete_training",
    )
    for i, rec in enumerate(runs):
        rec["costs"] = {
            stage: budget["ledger"][f"{i}/{stage}"]["used"]
            for stage in ("train", "gradient", "importance", "curvature")
        }
    pointers = lambda name: [f"runs/{i}/{name}" for i in range(4)]
    missing = [
        f"runs/{i}/{name}"
        for i, r in enumerate(runs)
        for name in ("gradient", "importance", "curvature", "drift")
        if r[name]["status"] != "ok"
    ]
    return finalize(
        dict(
            schema_version="vdp/0.1",
            protocol_revision="task002/1",
            source_commit=source,
            target_instance_id=target_id,
            dimension=d,
            dtype="float64",
            transform="identity",
            environment=dict(
                python=platform.python_version(),
                numpy=np.__version__,
                device="cpu",
                backend="numpy",
            ),
            configuration=dict(
                **asdict(config), reference_location=base, reference_scale=scale
            ),
            rng_manifest=manifest,
            budget=budget,
            runs=runs,
            profile=dict(
                optimization=dict(
                    rhat=rh,
                    drift=pointers("drift"),
                    stationarity=pointers("gradient"),
                    agreement=dict(
                        matrix=matrix,
                        pairs=pairs,
                        median=np.median(vals) if valid else None,
                        maximum=max(vals) if valid else None,
                    ),
                ),
                dependence=pointers("curvature"),
                geometry=pointers("curvature"),
                tails=pointers("importance"),
                possible_basins=metric(
                    None,
                    "basins",
                    4,
                    "inconclusive",
                    warnings=["see agreement, ELBO, drift and gradient"],
                ),
            ),
            evidence=dict(
                missing_or_flagged=missing,
                recommendation="abstain",
                warnings=[
                    "Four starts can miss modes. Optimizer iterates are not posterior draws.",
                    "Coordinate SEs assume finite variance. Two-way bootstrap percentiles are exploratory.",
                    "One importance batch per fit. No Pareto shape interval.",
                ],
            ),
        )
    ), trajectories
