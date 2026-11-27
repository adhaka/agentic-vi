"""Run exactly the eight frozen VDP profiles into a fresh output directory."""

import argparse
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

from agentic_vi.vdp.budget import Budget, autodiff_backend
from agentic_vi.vdp.probe import Config, profile
from .target import density

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "research/task002"))
from research.task002.study import archive, artifact, save

HORIZONS = (250, 500, 1000, 3000)
FORMS = ("centered", "noncentered")


def run(root):
    root.mkdir(parents=True, exist_ok=False)
    (root / "arrays").mkdir()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    start = time.perf_counter()
    cells = []
    for arm, form in enumerate(FORMS):
        for steps in HORIZONS:
            c = Config(steps)
            B = c.full_cost()
            meter = Budget(B, 10, autodiff_backend(density(form)), c.allocations(B))
            p, trajs = profile(
                10,
                c,
                meter.scope,
                meter.report,
                target_id=f"instance-{arm}",
                source=source,
            )
            p["protocol_revision"] = "task003/1"
            p["environment"].update(
                target_backend="torch autodiff",
                torch=str(torch.__version__),
                platform=platform.platform(),
                threads=torch.get_num_threads(),
            )
            # Probe operates in native unconstrained coordinates; model Jacobians are inside density.
            for r, traj in zip(p["runs"], trajs, strict=True):
                r["trajectory"] = archive(root, traj)
                for group, keys in [
                    ("importance", ["smoothed_log_weights"]),
                    ("curvature", ["directional", "estimated_diagonals"]),
                ]:
                    values = r[group]["value"]
                    for key in keys:
                        if values and values.get(key) is not None:
                            values[key] = artifact(root, values[key])
            name = f"{form}-{steps}"
            save(root / (name + ".json"), p)
            cells.append(
                dict(
                    profile=name, form=form, steps=steps, B=B, used=p["budget"]["used"]
                )
            )
            save(
                root / "index.json",
                dict(cells=cells, elapsed=time.perf_counter() - start),
            )
            print(name, p["budget"]["stop_reason"], p["budget"]["used"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    run(parser.parse_args().output)
