"""Produce one row per fit for review. No target primitives are called."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(root):
    rows = []
    for cell in json.loads((root / "index.json").read_text())["cells"]:
        folder = root / cell["name"]
        p = json.loads((folder / "profile.json").read_text())
        evaluator = json.loads((folder / "evaluator.json").read_text())
        opt = p["profile"]["optimization"]
        rh = opt["rhat"]["value"]
        for r, e in zip(p["runs"], evaluator["runs"], strict=True):
            g = r["gradient"]
            w = r["importance"]
            c = r["curvature"]
            drift = r["drift"]["value"]
            row = dict(
                cell=cell["name"],
                target=evaluator["kind"],
                d=p["dimension"],
                arm=cell["arm"],
                run=r["id"],
                steps=r["steps"],
                B=cell["B"],
                used=cell["used"],
                status=r["status"],
                seconds=r["seconds"],
                **e,
                rhat_max=None if rh is None else rh["maximum"]["quantiles"][-1],
                pair_max=opt["agreement"]["maximum"],
                pair_median=opt["agreement"]["median"],
                mu_drift=drift["mu"]["rms"] if drift else None,
                lambda_drift=drift["lambda"]["rms"] if drift else None,
                grad=g["value"]["rms"],
                noise=g["uncertainty"]["noise_rms"],
                dependence=c["value"]["dependence"],
                geometry=c["value"]["geometry"],
                dep_lo=c["uncertainty"]["percentiles"][0][0],
                dep_hi=c["uncertainty"]["percentiles"][1][0],
                ess=w["value"]["raw_ess"],
                psis_ess=w["value"]["psis_ess"],
                k=w["value"]["k_hat"],
                elbo=w["value"]["elbo"],
                elbo_se=w["uncertainty"]["elbo_se"],
            )
            rows.append(row)
    return rows


def plot(rows, output):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), layout="constrained")
    for kind, color in [
        ("independent", "#246d9f"),
        ("correlated", "#b74b1a"),
        ("ill_conditioned", "#39744b"),
    ]:
        group = [
            r
            for r in rows
            if r["target"] == kind and r["d"] == 100 and r["arm"] == "standard"
        ]
        for ax, key in zip(axes, ["gap_per_d", "dependence", "k"]):
            x = sorted({r["steps"] for r in group})
            y = np.array([[r[key] for r in group if r["steps"] == t] for t in x])
            ax.plot(
                x, np.median(y, axis=1), "o-", color=color, label=kind.replace("_", " ")
            )
            ax.fill_between(x, y.min(axis=1), y.max(axis=1), color=color, alpha=0.13)
    for ax, title in zip(
        axes,
        [
            "Excess reverse KL per dimension",
            "Standardized curvature interaction",
            "Pareto shape estimate",
        ],
    ):
        ax.set_xscale("log")
        ax.set_xticks([250, 500, 1000, 3000], labels=["250", "500", "1000", "3000"])
        ax.set_xlabel("Updates per fit")
        ax.set_title(title, fontsize=10)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Nats")
    axes[1].set_ylabel("Energy ratio (square root)")
    axes[2].set_ylabel("Pareto k")
    axes[0].legend(fontsize=8)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--figure", type=Path)
    args = parser.parse_args()
    rows = summarize(args.results)
    with args.csv.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.figure:
        plot(rows, args.figure)
