from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import unittest

import arviz as az
import numpy as np
from numpy.testing import assert_allclose
import torch

from agentic_vi.vdp.budget import Budget, Denied, autodiff_backend
from agentic_vi.vdp.diagnostics import (
    agreement,
    curvature,
    drift,
    energies,
    importance,
    rhat,
)
from agentic_vi.vdp.probe import Config, fit, pathwise, profile, streams

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/task002"))
from benchmark import evaluate, seeded_rotation, target_backend
from study import convert, design


def small(offset=0.0, limit=None, hvp=True, backend=None):
    c = Config(80, batch=4, G=12, S=64, J=3, R=4, bootstrap=8)
    budget = c.full_cost() if limit is None else limit
    provider = (
        target_backend(np.zeros(2), np.eye(2), offset=offset)
        if backend is None
        else backend
    )
    meter = Budget(budget, 2, provider, c.allocations(budget))
    return profile(
        2, c, meter.scope, meter.report, target_id="opaque", source="test", hvp=hvp
    )


class MeterTests(unittest.TestCase):
    def test_primitive_counts_boundary_and_failures(self):
        calls = []
        backend = target_backend(np.zeros(2), np.eye(2))

        def spy(order, x, v):
            calls.append((order, x.size // 2))
            if x.max() > 2:
                raise ValueError("attempted failure")
            return backend(order, x, v)

        meter = Budget(12, 2, spy, {"a": 12})
        api = meter.scope("a")
        self.assertFalse(hasattr(api, "precision") or hasattr(api, "_backend"))
        x = np.ones(2)
        api.value(x)
        api.value_grad(np.ones((2, 2)))
        api.value_grad_hvp(np.ones((2, 2)), np.ones((2, 2)))
        with self.assertRaises(ValueError):
            api.value(x * 3)
        with self.assertRaises(Denied):
            api.value(x)
        q = meter.report()
        self.assertEqual((q["f"], q["g"], q["h"], q["remaining"]), (6, 4, 2, 0))
        self.assertEqual(
            (q["attempts"], q["completed"], q["failed"], q["denied"]), (5, 3, 1, 1)
        )
        self.assertEqual(sum(k * n for k, n in calls), 12)
        with self.assertRaises(ValueError):
            api.value(np.empty((0, 2)))
        with self.assertRaises(ValueError):
            Budget(1, 2, spy, {"a": -1})

    def test_matrix_free_autodiff_backend(self):
        # Varying diagonal curvature plus a constant cross-coordinate interaction.
        fn = lambda x: -0.25 * (x**4).sum(dim=-1) - x[..., 0] * x[..., 1]
        x = np.array([[0.2, -1.0], [2.0, 0.7]])
        v = np.array([[1.0, -0.2], [-0.3, 0.6]])
        api = Budget(6, 2, autodiff_backend(fn), {"hvp": 6}).scope("hvp")
        value, g, hv = api.value_grad_hvp(x, v)
        assert_allclose(value, -0.25 * (x**4).sum(axis=-1) - x[:, 0] * x[:, 1])
        assert_allclose(g, -(x**3) - x[:, ::-1])
        assert_allclose(hv, 3 * x**2 * v + v[:, ::-1])
        linear = autodiff_backend(lambda x: x.sum(dim=-1))
        assert_allclose(linear(3, x, v)[2], np.zeros_like(x))

    def test_concurrent_reservation(self):
        api = Budget(7, 1, target_backend(np.zeros(1), np.eye(1)), {"a": 7}).scope("a")

        def call(i):
            try:
                api.value(np.ones(1))
                return 1
            except Denied:
                return 0

        with ThreadPoolExecutor(4) as pool:
            self.assertEqual(sum(pool.map(call, range(50))), 7)

    def test_training_denial_preserves_parameters(self):
        api = Budget(1, 2, target_backend(np.zeros(2), np.eye(2)), {"a": 1}).scope("a")
        p = np.zeros((2, 2))
        final, traj, status, _ = fit(api, p, Config(10), np.random.default_rng(12))
        assert_allclose(final, p)
        self.assertEqual((status, len(traj["raw_states"])), ("insufficient_budget", 1))


class InferenceTests(unittest.TestCase):
    def test_pathwise_autodiff_and_clamp(self):
        raw = torch.tensor(
            [-13.0, -12.0, -0.3, 8.0, 9.0], dtype=torch.float64, requires_grad=True
        )
        mu = torch.arange(5, dtype=torch.float64, requires_grad=True)
        eps = torch.tensor(np.random.default_rng(2).normal(size=(64, 5)))
        lam = raw.clamp(-12, 8)
        x = mu + lam.exp() * eps
        q = torch.distributions.Normal(mu, lam.exp())
        obj = (-0.5 * x.square() - q.log_prob(x)).sum(axis=1).mean()
        grads = torch.autograd.grad(obj, (mu, raw))
        assert_allclose(
            pathwise(-x.detach().numpy(), eps.numpy(), raw.detach().numpy()),
            np.stack([g.numpy() for g in grads]),
            atol=1e-8,
            rtol=1e-10,
        )

    def test_shifts_reproducibility_and_ledger(self):
        a, ta = small()
        b, tb = small(19.0)
        self.assertEqual(a["rng_manifest"], b["rng_manifest"])
        self.assertEqual(
            len(set(s for row in a["rng_manifest"] for s in row.values())), 24
        )
        self.assertEqual(streams(0, 80)[0]["train"], streams(0, 90)[0]["train"])
        self.assertNotEqual(
            streams(0, 80)[0]["importance"], streams(0, 90)[0]["importance"]
        )
        for ra, rb, x, y in zip(a["runs"], b["runs"], ta, tb):
            self.assertEqual(ra["status"], "ok")
            assert_allclose(x["raw_states"], y["raw_states"])
            assert_allclose(x["preupdate_objective"] + 19, y["preupdate_objective"])
            assert_allclose(ra["end"]["mu"], x["raw_states"][-1, 0])
            assert_allclose(
                x["raw_states"][1] - x["raw_states"][0],
                0.03
                * x["preupdate_gradient"][0]
                / (abs(x["preupdate_gradient"][0]) + 1e-8),
            )
            ia, ib = (r["importance"]["value"] for r in (ra, rb))
            assert_allclose(ia["elbo"] + 19, ib["elbo"])
            for key in ("raw_ess", "k_hat", "relative_ess", "smoothed_log_weights"):
                assert_allclose(ia[key], ib[key], atol=1e-10)
            seeds = a["rng_manifest"][ra["id"]]
            eps = np.random.default_rng(seeds["importance"]).normal(size=(64, 2))
            end = ra["end"]
            la = end["effective_lambda"]
            p = end["mu"] + eps * np.exp(la)
            logw = -0.5 * (p * p - eps * eps - np.log(2 * np.pi)).sum(axis=1) + la.sum()
            assert_allclose(ia["elbo"], logw.mean())
        self.assertEqual(a["budget"]["B"], a["budget"]["used"])
        json.dumps(a, default=convert, allow_nan=False)

    def test_partial_failed_and_unsupported(self):
        p, _ = small(limit=0)
        self.assertEqual(p["budget"]["used"], 0)
        self.assertTrue(
            all(r["importance"]["status"] == "insufficient_budget" for r in p["runs"])
        )
        self.assertEqual(
            p["profile"]["optimization"]["rhat"]["status"], "insufficient_samples"
        )
        p, _ = small(hvp=False)
        self.assertEqual(p["budget"]["h"], 0)
        self.assertTrue(
            all(r["curvature"]["status"] == "unsupported" for r in p["runs"])
        )

        def invalid(order, x, v):
            raise RuntimeError("invalid gradient")

        p, _ = small(backend=invalid)
        self.assertTrue(all(r["status"] == "run_failed" for r in p["runs"]))
        self.assertEqual(p["budget"]["failed"], 4)
        self.assertEqual(p["profile"]["optimization"]["rhat"]["status"], "run_failed")
        json.dumps(p, default=convert, allow_nan=False)

    def test_oracle_and_matrix(self):
        cov = seeded_rotation(5, 8)
        assert_allclose(np.linalg.eigvalsh(cov), np.geomspace(0.001, 1, 5))
        assert_allclose(cov, seeded_rotation(5, 8))
        opt = 1 / np.linalg.inv(cov).diagonal()
        e = evaluate(
            dict(mu=np.zeros(5), effective_lambda=np.log(opt) / 2), np.zeros(5), cov
        )
        self.assertAlmostEqual(e["gap"], 0.0)
        self.assertGreater(e["forward_kl"], 0)
        self.assertEqual(sum(item[-1] for item in design(False)), 22446081)


class DiagnosticTests(unittest.TestCase):
    def test_reference_rhat(self):
        self.assertEqual(az.__version__, "0.21.0")
        rng = np.random.default_rng(2)
        x = rng.normal(size=(4, 101))
        for vals in (
            x,
            np.tile(x[0], (4, 1)),
            x + np.arange(4)[:, None] * 3,
            x * (1 + np.arange(4)[:, None]),
            x + np.arange(101) / 8,
            np.round(x),
        ):
            v = rhat(vals[:, :, None])["value"]
            for key, method in [
                ("classical", "identity"),
                ("split", "split"),
                ("rank", "z_scale"),
                ("folded", "folded"),
                ("maximum", "rank"),
            ]:
                assert_allclose(
                    v[key]["values"][0], float(az.rhat(vals, method=method)), rtol=1e-12
                )
        for vals, status in [
            (np.full_like(x, 0.1), "undefined"),
            (np.ones_like(x), "undefined"),
            (np.ones_like(x) * np.arange(4)[:, None], "separated_constant"),
        ]:
            self.assertEqual(
                rhat(vals[:, :, None])["value"]["classical"]["coordinate_status"],
                [status],
            )
        self.assertEqual(rhat(x[:3, :, None])["status"], "run_failed")
        self.assertEqual(rhat((x * np.nan)[:, :, None])["status"], "nonfinite")

    def test_drift_and_pairwise(self):
        a = np.array([1.0, -2.0])
        l = np.array([-10.0, 7.0])
        b = a + np.array([0.1, 3])
        k = l + 0.3
        p = agreement((a, l), (b, k))
        q = agreement((b, k), (a, l))
        self.assertEqual(agreement((a, l), (a, l))["joint"], 0)
        assert_allclose(p["joint"], q["joint"])
        assert_allclose(p["joint"], p["mean"] + p["scale"])
        expected = 0.25 * np.sum(
            np.exp(2 * (k - l))
            + np.exp(2 * (l - k))
            - 2
            + (a - b) ** 2 * (np.exp(-2 * k) + np.exp(-2 * l))
        )
        assert_allclose(p["joint"], expected)
        w = np.array([[0.0, 0.0], [0.0, 0.0], [20.0, 20.0], [1.0, 0.0], [1.0, 0.0]])
        self.assertEqual(drift(w)["value"]["mu"]["rms"], 1)

    def test_curvature_references(self):
        rng = np.random.default_rng(33)
        z = rng.choice([-1.0, 1.0], size=(4096, 2))
        for A in (
            np.diag([2.0, 1.0]),
            np.array([[1.0, -0.6], [-0.6, 1.0]]),
            np.array([[2.0, 0.4], [0.4, 0.7]]),
        ):
            u = np.broadcast_to(z @ A, (3, *z.shape))
            c = curvature(u, z, rng, 8)["value"]
            self.assertAlmostEqual(c["geometry"], 0, places=12)
            ref = np.linalg.norm(A - np.diag(A.diagonal())) / np.linalg.norm(A)
            self.assertAlmostEqual(c["dependence"], ref, delta=0.015)
        # Off-diagonal energy and total energy converge in repeated-probe averages.
        A = np.array([[2.0, 0.7], [0.7, 1.0]])
        errors = []
        for r in (4, 256):
            estimates = []
            for _ in range(200):
                z = rng.choice([-1.0, 1.0], size=(r, 2))
                f, o, _ = energies(np.tile(z @ A, (2, 1, 1)), z)
                estimates.append([f.mean(), o.mean()])
            err = np.asarray(estimates) - [np.sum(A * A), 0.98]
            assert_allclose(err.mean(axis=0), 0, atol=0.2)
            errors.append(np.mean(err**2))
        self.assertLess(errors[1], errors[0])
        # Re-express x in units scaled by a signed diagonal, transform probes consistently.
        s = np.array([-2.0, 0.3])
        sig = np.array([0.6, 1.4])
        original = sig[:, None] * A * sig
        transformed = abs(s * sig)[:, None] * (A / s[:, None] / s) * abs(s * sig)
        z2 = z * np.sign(s)
        c1 = curvature(np.tile(z @ original, (2, 1, 1)), z, np.random.default_rng(1), 8)
        c2 = curvature(
            np.tile(z2 @ transformed, (2, 1, 1)), z2, np.random.default_rng(1), 8
        )
        assert_allclose(c1["value"]["dependence"], c2["value"]["dependence"])
        assert_allclose(
            c1["uncertainty"]["percentiles"], c2["uncertainty"]["percentiles"]
        )

    def test_zero_indefinite_and_varying(self):
        rng = np.random.default_rng(11)
        z = rng.choice([-1.0, 1.0], size=(16, 2))
        u = np.stack([z, z * 2, z * 3])
        self.assertAlmostEqual(
            curvature(u, z, rng, 8)["value"]["geometry"], np.sqrt(1 / (14 / 3))
        )
        self.assertEqual(curvature(-u, z, rng, 8)["value"]["negative_fraction"], 1)
        self.assertEqual(curvature(u * 0, z, rng, 8)["status"], "degenerate")
        self.assertEqual(curvature(u[:1], z, rng, 8)["status"], "insufficient_samples")
        self.assertEqual(
            curvature(u[:, :1], z[:1], rng, 8)["status"], "insufficient_samples"
        )

    def test_psis(self):
        rng = np.random.default_rng(794)
        a = np.sort(rng.standard_t(5, size=1024))
        result = importance(a)["value"]
        assert_allclose(result["k_hat"], 0.6924625524537441)
        assert_allclose(result["psis_ess"], 150.9188994582971)
        w, k = az.psislw(a, reff=1)
        assert_allclose(result["smoothed_log_weights"], w)
        assert_allclose(result["k_hat"], k)
        self.assertEqual(importance(np.ones(32))["value"]["raw_ess"], 32)
        self.assertIsNone(importance(np.ones(32))["value"]["k_hat"])
        for bad in ([np.nan, 1], [1, np.inf], [-np.inf, -np.inf]):
            self.assertEqual(importance(np.array(bad))["status"], "nonfinite")
        zero = importance(np.r_[-np.inf, a])["value"]
        self.assertIsNone(zero["elbo"])
        self.assertEqual(zero["zero_weights"], 1)
        # N(0,.2²) proposal for N(0,1) produces highly variable weights.
        eps = rng.normal(size=4096)
        self.assertLess(
            importance(0.5 * (1 - 0.2**2) * eps**2)["value"]["relative_ess"], 0.5
        )
