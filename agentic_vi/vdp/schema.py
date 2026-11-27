"""Metric-specific views of shared estimators. Pure projection, no target access."""

from .diagnostics import metric


def measurement(group, field, units, status=None, uncertainty=None, count=None):
    payload = group["value"] or {}
    value = payload.get(field)
    return metric(
        value,
        units,
        group["sample_count"] if count is None else count,
        status or group["status"],
        uncertainty,
        group["warnings"],
    )


def finalize(record):
    profile = record["profile"]
    tails, interactions, geometries = [], [], []
    for run in record["runs"]:
        w = run["importance"]
        data = w["value"] or {}
        weights = {}
        for key, units in [
            ("raw_ess", "weight ESS"),
            ("relative_ess", "fraction"),
            ("max_weight", "fraction"),
            ("psis_ess", "smoothed weight ESS"),
            ("k_hat", "shape"),
            ("elbo", "unnormalized nats"),
        ]:
            status = w["status"]
            if data.get(key) is not None:
                status = "ok"
            elif key == "k_hat" and data.get("raw_ess") is not None:
                status = "degenerate"
            elif key == "elbo" and data.get("zero_weights", 0) > 0:
                status = "nonfinite"
            uncertainty = w["uncertainty"] if key == "elbo" else None
            weights[key] = measurement(w, key, units, status, uncertainty)
        tails.append(weights)
        c = run["curvature"]
        sensitivity = c["uncertainty"] or {}
        intervals = sensitivity.get("percentiles")
        for index, (key, collection) in enumerate(
            [("dependence", interactions), ("geometry", geometries)]
        ):
            uncertainty = None
            if intervals is not None:
                uncertainty = dict(
                    percentile_2_5=intervals[0][index],
                    percentile_97_5=intervals[1][index],
                    replicates=sensitivity["replicates"],
                    invalid=sensitivity["invalid"],
                )
            collection.append(
                measurement(
                    c, key, "energy ratio, square root", uncertainty=uncertainty
                )
            )
    profile.update(dependence=interactions, geometry=geometries, tails=tails)
    profile["possible_basins"]["warnings"] += [
        "Four-run agreement cannot exclude a shared missed basin."
    ]
    return record
