"""Regenerate the Task 003 radar figure from archived per-fit CSV (no inference calls).

Run: python research/task003/evaluation/plot_vdp_radar.py
Requires matplotlib and numpy (already used by Task 003).
"""
from pathlib import Path
import csv
from statistics import median

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
with (HERE / "per_fit.csv").open() as handle:
    rows = list(csv.DictReader(handle))

def summary(form):
    fits = [r for r in rows if r["form"] == form and int(r["steps"]) == 3000]
    assert len(fits) == 4
    return [
        max(float(r["rhat"]) for r in fits),
        median(float(r["dep"]) for r in fits),
        median(float(r["k"]) for r in fits),
        median(float(r["geometry"]) for r in fits),
        max(float(r["pair"]) for r in fits),
    ]

labels = ["Optimization\n(max R-hat)", "Dependence\n(curvature interaction)",
          "Tail / coverage\n(Pareto k)", "Geometry\n(curvature variation)",
          "Endpoint disagreement\n(max symmetrized KL)"]
angles = np.linspace(0, 2 * np.pi, 5, endpoint=False)
fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
ax.set_theta_zero_location("N")
ax.set_theta_direction(-1)
for form, name, color in [("centered", "Centered", "#2166ac"),
                           ("noncentered", "Non-centered", "#df7900")]:
    rh, dep, k, geom, pair = summary(form)
    # Axis-specific visualization scales, NOT calibrated difficulty scores.
    values = np.clip([(rh - 1) / 4, dep, k, geom, pair / 10], 0, 1)
    ax.plot(np.r_[angles, angles[0]], np.r_[values, values[0]],
            color=color, marker="o", lw=2.4, label=name)
    ax.fill(np.r_[angles, angles[0]], np.r_[values, values[0]], color=color, alpha=.13)
ax.set_xticks(angles, labels, fontsize=10)
ax.tick_params(axis="x", pad=25)
ax.set_ylim(0, 1)
ax.set_yticks([.25, .5, .75, 1])
ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=9)
ax.grid(alpha=.35)
fig.suptitle("Eight Schools | Variational Difficulty Profile", fontsize=18, weight="bold", y=.97)
fig.text(.5, .925, "3,000 Adam updates per run | four MFVI fits per parameterization",
         ha="center", fontsize=11)
ax.legend(loc="lower center", bbox_to_anchor=(.5, -.20), ncol=2, frameon=False)
fig.text(.5, .035, "Visualization only: (R-hat - 1)/4; curvature ratios and k unchanged; pairwise KL / 10 nats.\n"
         "All axes clipped to [0, 1]. Curvature and k: medians; R-hat and pairwise KL: maxima.\n"
         "Endpoint disagreement does not establish posterior multimodality.",
         ha="center", fontsize=9)
fig.subplots_adjust(left=.20, right=.80, top=.82, bottom=.22)
fig.savefig(HERE / "vdp_radar.png", dpi=220, bbox_inches="tight")
plt.close(fig)
