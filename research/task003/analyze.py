"""Audit frozen outputs, evaluate in scientific coordinates and render comparison figures."""

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from .reference import posterior, adaptive_evidence, fitted
from .study import HORIZONS, FORMS, save


def array(root, ref):
    data = gzip.decompress((root / ref["path"]).read_bytes())
    assert hashlib.sha256(data).hexdigest() == ref["sha256_npy"]
    a = np.load(io.BytesIO(data), allow_pickle=False)
    assert list(a.shape) == ref["shape"] and str(a.dtype) == ref["dtype"]
    return a


def audit(root, cell, p):
    t = cell["steps"]
    cost = p["budget"]
    assert (cost["f"], cost["g"], cost["h"]) == (
        4 * (64 * t + 256 + 1024 + 128),
        4 * (64 * t + 256 + 128),
        512,
    )
    assert cost["B"] == cell["B"] == cost["used"] == cell["used"] == 512 * t + 7680
    assert cost["remaining"] == cost["failed"] == cost["denied"] == 0
    assert cost["attempts"] == cost["completed"]
    for key in ("f", "g", "h", "used", "attempts", "completed"):
        assert sum(row[key] for row in cost["ledger"].values()) == cost[key]
    assert (
        len(p["runs"]) == 4 and p["profile"]["optimization"]["rhat"]["status"] == "ok"
    )
    for r in p["runs"]:
        assert r["steps"] == t and r["status"] == "ok"
        tr = np.concatenate([array(root, ref) for ref in r["trajectory"]["chunks"]])
        assert tr.shape == (t, 41) and np.isfinite(tr).all()
        assert_allclose(tr[-1, :10], r["end"]["mu"])
        assert_allclose(tr[-1, 10:20], r["end"]["raw_lambda"])
        w = array(root, r["importance"]["value"]["smoothed_log_weights"])
        assert_allclose(np.exp(w).sum(), 1)
        for key in ("estimated_diagonals", "directional"):
            array(root, r["curvature"]["value"][key])


def comparison_figure(rows, specs, file):
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.4), layout="constrained")
    for ax, (key, title, unit) in zip(axes.flat, specs):
        for form, label, color in [
            ("centered", "Centered", "#97431c"),
            ("noncentered", "Non-centered", "#215e82"),
        ]:
            vals = np.array(
                [
                    [r[key] for r in rows if r["form"] == form and r["steps"] == t]
                    for t in HORIZONS
                ]
            )
            ax.plot(HORIZONS, np.median(vals, axis=1), "o-", color=color, label=label)
            ax.fill_between(
                HORIZONS, vals.min(axis=1), vals.max(axis=1), color=color, alpha=0.16
            )
        ax.set(xscale="log", title=title, ylabel=unit, xlabel="Adam updates per fit")
        ax.set_xticks(HORIZONS, labels=["250", "500", "1000", "3000"])
        ax.grid(axis="y", alpha=0.2)
    axes.flat[0].legend()
    fig.savefig(file, dpi=160)
    plt.close(fig)


def analyze(root, out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    truth = posterior(256)
    coarse = posterior(128)
    alt, estimated_rel = adaptive_evidence()
    differences = [
        abs(v - coarse["marginals"][k][s])
        for k, vals in truth["marginals"].items()
        for s, v in vals.items()
    ]
    truth["integration_check"] = dict(
        orders=[128, 256],
        max_absolute_marginal_change=max(differences),
        log_evidence_change=truth["log_evidence"] - coarse["log_evidence"],
        adaptive_dense_log_evidence=alt,
        adaptive_relative_error_estimate=estimated_rel,
    )
    assert max(differences) < 1e-9 and abs(alt - truth["log_evidence"]) < 1e-10
    save(out / "reference.json", truth)
    index = json.loads((root / "index.json").read_text())
    cells = index["cells"]
    assert [(c["form"], c["steps"]) for c in cells] == [
        (f, t) for f in FORMS for t in HORIZONS
    ]
    rows = []
    for c in cells:
        p = json.loads((root / (c["profile"] + ".json")).read_text())
        audit(root, c, p)
        evals = []
        opt = p["profile"]["optimization"]
        for r in p["runs"]:
            e = fitted(c["form"], r["end"], truth)
            evals.append(dict(run=r["id"], **e))
            w = r["importance"]["value"]
            h = r["curvature"]["value"]
            err = e["errors"]
            rows.append(
                dict(
                    form=c["form"],
                    steps=c["steps"],
                    run=r["id"],
                    used=c["used"],
                    kl=e["reverse_kl"],
                    elbo=e["elbo"],
                    mc_elbo=w["elbo"],
                    mc_elbo_se=r["importance"]["uncertainty"]["elbo_se"],
                    mu_mean_error=err["mu"]["mean"],
                    mu_sd_error=err["mu"]["sd"],
                    tau_mean_error=err["tau"]["mean"],
                    tau_sd_error=err["tau"]["sd"],
                    tau_q05_error=err["tau"]["q05"],
                    tau_q50_error=err["tau"]["q50"],
                    tau_q95_error=err["tau"]["q95"],
                    **{
                        "school_" + stat + "_mae": np.mean(
                            [abs(err[f"theta_{i}"][stat]) for i in range(1, 9)]
                        )
                        for stat in ("mean", "sd", "q05", "q50", "q95")
                    },
                    geometry=h["geometry"],
                    dep=h["dependence"],
                    negative=h["negative_fraction"],
                    k=w["k_hat"],
                    ess=w["raw_ess"],
                    psis_ess=w["psis_ess"],
                    grad=r["gradient"]["value"]["rms"],
                    noise=r["gradient"]["uncertainty"]["noise_rms"],
                    drift=r["drift"]["value"]["mu"]["rms"],
                    pair=opt["agreement"]["maximum"],
                    rhat=opt["rhat"]["value"]["maximum"]["quantiles"][-1],
                )
            )
        save(out / (c["profile"] + ".json"), dict(profile=c["profile"], runs=evals))
    with (out / "per_fit.csv").open("w") as fp:
        writer = csv.DictWriter(fp, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    comparison_figure(
        rows,
        [
            ("geometry", "Curvature variation", "Ratio"),
            ("dep", "Local curvature interaction", "Ratio"),
            ("pair", "Largest endpoint distance", "Symmetrized KL (nats)"),
            ("k", "Importance tail shape", "Pareto k"),
        ],
        out / "diagnostics.png",
    )
    comparison_figure(
        rows,
        [
            ("kl", "Joint posterior error", "Reverse KL (nats)"),
            ("tau_q95_error", "Upper tail of τ", "Signed 95th-percentile error"),
            ("mu_sd_error", "Uncertainty in μ", "Signed SD error"),
            (
                "school_sd_mae",
                "Uncertainty in school effects",
                "Mean absolute SD error",
            ),
        ],
        out / "accuracy.png",
    )
    save(
        out / "audit.json",
        dict(
            profiles=len(cells),
            fits=len(rows),
            inference_teu=sum(c["used"] for c in cells),
            audit_evaluation_plot_seconds=time.perf_counter() - start,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.results, args.output)
