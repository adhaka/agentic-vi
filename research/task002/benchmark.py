"""Privileged benchmark construction and post-fit oracle evaluation."""

from pathlib import Path

import numpy as np


def seeded_rotation(d, seed):
    """v1 sign-fixed QR of an iid Gaussian matrix; return the realized covariance."""
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
    q *= np.where(np.diag(r) < 0, -1.0, 1.0)
    return (q * np.geomspace(1.0, 0.001, d)) @ q.T


def load_instance(kind, d):
    root = Path(__file__).resolve().parents[1] / "task001/results"
    return np.linspace(-0.5, 0.5, d), np.load(
        root / f"{kind}_d{d}_cov.npy", allow_pickle=False
    )


def target_backend(mean, cov, offset=0.0):
    precision = np.linalg.solve(cov, np.eye(len(mean)))
    precision = (precision + precision.T) / 2

    def primitive(order, x, v):
        delta = x - mean
        action = delta @ precision
        val = offset - 0.5 * (action * delta).sum(axis=-1)
        if order == 1:
            return (val,)
        if order == 2:
            return val, -action
        return val, -action, v @ precision

    return primitive


def evaluate(end, mean, cov):
    d = len(mean)
    delta = end["mu"] - mean
    var = np.exp(2 * end["effective_lambda"])
    chol = np.linalg.cholesky(cov)
    diag_p = np.square(np.linalg.solve(chol, np.eye(d))).sum(axis=0)
    logdet = 2 * np.log(chol.diagonal()).sum()
    rev = 0.5 * (
        diag_p @ var
        + np.sum(np.linalg.solve(chol, delta) ** 2)
        - d
        + logdet
        - np.log(var).sum()
    )
    floor = 0.5 * (logdet + np.log(diag_p).sum())
    fwd = 0.5 * (
        np.sum((cov.diagonal() + delta**2) / var) - d + np.log(var).sum() - logdet
    )
    sd = np.sqrt(cov.diagonal())
    corr = cov / sd[:, None] / sd
    return dict(
        reverse_kl=rev,
        forward_kl=fwd,
        floor=floor,
        gap=rev - floor,
        reverse_per_d=rev / d,
        forward_per_d=fwd / d,
        floor_per_d=floor / d,
        gap_per_d=(rev - floor) / d,
        mean_mse=np.mean(delta**2),
        variance_mse=np.mean((var - sd**2) ** 2),
        optimal_variance_mse=np.mean((var - 1 / diag_p) ** 2),
        min_var_ratio=min(var / sd**2),
        covariance_fro=np.linalg.norm(np.diag(var) - cov),
        correlation_fro=np.linalg.norm(np.eye(d) - corr),
        covariance_rmse=np.linalg.norm(np.diag(var) - cov) / d,
        correlation_rmse=np.linalg.norm(np.eye(d) - corr) / d,
        exact_unnormalized_elbo=0.5 * (d * np.log(2 * np.pi) + logdet) - rev,
    )
