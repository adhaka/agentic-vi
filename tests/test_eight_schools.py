import unittest
import numpy as np
import torch
from numpy.testing import assert_allclose
from scipy.special import ndtri
from scipy.stats import halfcauchy, norm, qmc
from agentic_vi.vdp.budget import autodiff_backend
from research.task003.data import Y, SE
from research.task003.target import density
from research.task003.reference import conditional, fitted, posterior, adaptive_evidence


def independent_joint(form, x):
    mu, eta, w = x[..., 0], x[..., 1], x[..., 2:]
    t = np.exp(eta)
    theta = w if form == "centered" else mu[..., None] + t[..., None] * w
    prior = norm.logpdf(theta, mu[..., None], t[..., None]).sum(-1)
    return (
        norm.logpdf(mu, scale=5)
        + halfcauchy.logpdf(t, scale=5)
        + eta
        + prior
        + (8 * eta if form == "noncentered" else 0)
        + norm.logpdf(Y, theta, SE).sum(-1)
    )


class SchoolsTests(unittest.TestCase):
    def test_normalization_jacobian_and_derivatives(self):
        rng = np.random.default_rng(4)
        for eta in (-2.0, 0.0, 1.8):
            x = rng.normal(size=10)
            x[1] = eta
            centered = x.copy()
            centered[2:] = x[0] + np.exp(eta) * x[2:]
            a = density("centered")(torch.tensor(centered)).item()
            b = density("noncentered")(torch.tensor(x)).item()
            assert_allclose(b, a + 8 * eta, atol=1e-12)
            for form in ("centered", "noncentered"):
                v = rng.normal(size=10)
                value, grad, hvp = autodiff_backend(density(form))(3, x, v)
                assert_allclose(value, independent_joint(form, x), atol=1e-11)

                def fd(point, h=2e-5):
                    delta = np.eye(10) * h
                    return (
                        independent_joint(form, point + delta)
                        - independent_joint(form, point - delta)
                    ) / (2 * h)

                assert_allclose(grad, fd(x), rtol=2e-7, atol=2e-8)
                h = 1e-3

                def direction(step):
                    return -(fd(x + step * v) - fd(x - step * v)) / (2 * step)

                assert_allclose(
                    hvp, (4 * direction(h / 2) - direction(h)) / 3, rtol=2e-5, atol=1e-5
                )

    def test_conditionals(self):
        # Dense Gaussian conditioning on y, independent of the shrinkage formulas.
        for tau in (0.01, 4.0, 40.0):
            K = 25 * np.ones((9, 9)) + np.diag([0.0, *[tau * tau] * 8])
            obs = K[1:, 1:] + np.diag(np.square(SE))
            m = K[:, 1:] @ np.linalg.solve(obs, Y)
            cov = K - K[:, 1:] @ np.linalg.solve(obs, K[1:, :])
            _, means, vs = conditional(tau)
            assert_allclose(means[0], m)
            assert_allclose(vs[0], cov.diagonal())

    def test_quadrature(self):
        a, b = posterior(128), posterior(256)
        assert_allclose(a["log_evidence"], adaptive_evidence()[0], atol=1e-10, rtol=0)
        for key in a["marginals"]:
            assert_allclose(
                list(a["marginals"][key].values()),
                list(b["marginals"][key].values()),
                rtol=1e-9,
                atol=1e-10,
            )

    def test_fit_evaluator_independent_integral(self):
        truth = posterior(128)
        loc = np.linspace(-0.3, 0.7, 10)
        sd = np.linspace(0.5, 1.2, 10)
        eps = ndtri(qmc.Sobol(10, scramble=True, seed=409).random_base2(16))
        x = loc + sd * eps
        for form in ("centered", "noncentered"):
            e = fitted(form, dict(mu=loc, effective_lambda=np.log(sd)), truth)
            kl = truth["log_evidence"] + np.mean(
                norm.logpdf(x, loc, sd).sum(1) - independent_joint(form, x)
            )
            assert_allclose(e["reverse_kl"], kl, atol=0.02)
            theta = (
                x[:, 2:]
                if form == "centered"
                else x[:, 0, None] + np.exp(x[:, 1, None]) * x[:, 2:]
            )
            stats = e["marginals"]["theta_8"]
            assert_allclose(
                [stats[k] for k in ("mean", "sd", "q05", "q50", "q95")],
                [
                    theta[:, -1].mean(),
                    theta[:, -1].std(),
                    *np.quantile(theta[:, -1], [0.05, 0.5, 0.95]),
                ],
                atol=0.025,
            )
