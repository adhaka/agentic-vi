"""Execute the frozen development matrix, or the separately charged smoke profile."""

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agentic_vi.vdp.budget import Budget
from agentic_vi.vdp.probe import Config, profile
from benchmark import evaluate, load_instance, target_backend


def convert(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def save(path, data):
    path.write_text(
        json.dumps(data, default=convert, allow_nan=False, separators=(",", ":")) + "\n"
    )


def artifact(root, array):
    array = np.asarray(array)
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    raw = stream.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    path = Path("arrays") / (digest + ".npy.gz")
    dest = root / path
    if not dest.exists():
        dest.write_bytes(gzip.compress(raw, compresslevel=1, mtime=0))
    return dict(
        path=str(path),
        sha256_npy=digest,
        shape=list(array.shape),
        dtype=str(array.dtype),
    )


def archive(root, traj):
    states = traj["raw_states"]
    # Each row is a post-update state with the gradient/objective that produced it.
    # Effective log scales are exactly clip(raw,-12,8), including saturation.
    rows = np.column_stack(
        [
            states[1:].reshape(len(states) - 1, states.shape[1] * states.shape[2]),
            traj["preupdate_gradient"].reshape(
                len(states) - 1, states.shape[1] * states.shape[2]
            ),
            traj["preupdate_objective"],
        ]
    )
    return dict(
        initial=states[0],
        chunks=[artifact(root, rows[k : k + 250]) for k in range(0, len(rows), 250)],
        columns="mu[d], raw_lambda[d], ascent_mu[d], ascent_raw_lambda[d], preupdate_objective",
        effective_lambda="clip(raw_lambda,-12,8)",
        lr=traj["lr"],
        teu_per_step=traj["teu_per_step"],
    )


def design(smoke):
    if smoke:
        c = Config(80, batch=8, G=16, S=64, J=3, R=4, bootstrap=20)
        yield "smoke", "independent", 10, "instance-00", c, c.full_cost()
        return
    for k, kind in enumerate(("independent", "correlated", "ill_conditioned")):
        for j, d in enumerate((10, 50, 100)):
            for t in (250, 500, 1000, 3000):
                c = Config(t)
                yield "standard", kind, d, f"instance-{3 * k + j:02d}", c, c.full_cost()
    for name in ("short", "unstable", "common"):
        c = Config(
            250, lr=0.6 if name == "unstable" else 0.03, common_start=name == "common"
        )
        yield (
            name,
            "correlated",
            10,
            "instance-03",
            c,
            10241 if name == "short" else c.full_cost(),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    (root / "arrays").mkdir()
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    start = time.perf_counter()
    index = []
    for arm, kind, d, target_id, c, limit in design(args.smoke):
        name = f"{arm}-{target_id}-{c.steps}"
        folder = root / name
        folder.mkdir()
        mean, cov = load_instance(kind, d)
        meter = Budget(limit, d, target_backend(mean, cov), c.allocations(limit))
        result, trajs = profile(
            d, c, meter.scope, meter.report, target_id=target_id, source=source
        )
        for rec, traj in zip(result["runs"], trajs):
            rec["trajectory"] = archive(root, traj)
            for stage, keys in [
                ("importance", ["smoothed_log_weights"]),
                ("curvature", ["directional", "estimated_diagonals"]),
            ]:
                data = rec[stage]["value"]
                for key in keys:
                    if data is not None and data.get(key) is not None:
                        data[key] = artifact(root, data[key])
        save(folder / "profile.json", result)
        tic = time.perf_counter()
        truth = [evaluate(r["end"], mean, cov) for r in result["runs"]]
        save(
            folder / "evaluator.json",
            dict(
                kind=kind,
                dimension=d,
                target_id=target_id,
                runs=truth,
                covariance_sha256=hashlib.sha256(cov.tobytes()).hexdigest(),
                seconds=time.perf_counter() - tic,
                inference_teu=0,
            ),
        )
        index.append(dict(name=name, arm=arm, B=limit, used=result["budget"]["used"]))
        save(
            root / "index.json", dict(cells=index, elapsed=time.perf_counter() - start)
        )
        print(name, result["budget"]["used"], flush=True)


if __name__ == "__main__":
    main()
