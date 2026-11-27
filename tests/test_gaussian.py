import math
import unittest

import torch
from torch.distributions import MultivariateNormal, kl_divergence

from agentic_vi.algorithms.mfvi import MFVIConfig, MeanFieldGaussianVI
from agentic_vi.evaluation import evaluate_mfvi
from agentic_vi.targets import GaussianTarget, make_gaussian_target


class GaussianValidation(unittest.TestCase):
    def metrics(self, target, mean, variance):
        model = MeanFieldGaussianVI(MFVIConfig(dimension=mean.numel()))
        with torch.no_grad():
            model.mu.copy_(mean)
            model.log_std.copy_(variance.log() / 2)
        return evaluate_mfvi(model, target)

    def test_bivariate_optimum_and_perturbation(self):
        # Conditional variances from the bivariate conditional density.
        cov = torch.tensor([[4., 1.], [1., 1.]], dtype=torch.float64)
        mean = torch.tensor([1., -2.], dtype=torch.float64)
        optimum = torch.tensor([3., .75], dtype=torch.float64)
        target = GaussianTarget(mean, cov)
        floor = -0.5 * math.log(.75)
        for shift, scale in [(0., 1.), (.4, 2.)]:
            q = MultivariateNormal(mean + shift, torch.diag(optimum * scale))
            p = MultivariateNormal(mean, cov)
            ref = float(kl_divergence(q, p))
            m = self.metrics(target, q.mean, q.variance)
            self.assertAlmostEqual(m['exact_reverse_kl_q_to_p'], ref)
            self.assertAlmostEqual(m['irreducible_mean_field_kl'], floor)
            self.assertAlmostEqual(m['optimization_gap_kl'], ref - floor)
            self.assertAlmostEqual(m['final_elbo'], -ref)
            self.assertAlmostEqual(m['posterior_mean_mse'], shift**2)
            if shift == 0:
                self.assertAlmostEqual(m['optimal_mean_field_variance_mse'], 0)
                self.assertAlmostEqual(m['posterior_marginal_variance_mse'], (1 + .25**2)/2)

    def test_independent_exact_recovery(self):
        target = make_gaussian_target(10, 'independent')
        metrics = self.metrics(target, target.mean, target.covariance.diagonal())
        for name in ('exact_reverse_kl_q_to_p', 'irreducible_mean_field_kl',
                     'optimization_gap_kl', 'posterior_marginal_variance_mse'):
            self.assertAlmostEqual(metrics[name], 0)

    def test_target_density_and_spd(self):
        for kind in ('independent', 'correlated', 'ill_conditioned'):
            for dimension in (10, 50, 100):
                with self.subTest(kind=kind, dimension=dimension):
                    target = make_gaussian_target(dimension, kind)
                    reference = MultivariateNormal(target.mean, target.covariance)
                    probes = torch.stack([target.mean, target.mean.roll(1)])
                    torch.testing.assert_close(target.log_prob(probes), reference.log_prob(probes))

    def test_runner_evaluates_post_update_distribution(self):
        from argparse import Namespace
        from run_baseline import run_one
        args = Namespace(dtype='float64', steps=1, learning_rate=.03, mc_samples=128,
                         dimension=2, log_every=1, target='independent')
        result = run_one(args, 0)
        q = MultivariateNormal(torch.tensor(result['fitted_mean'], dtype=torch.float64),
                               torch.diag(torch.tensor(result['fitted_variance'], dtype=torch.float64)))
        target = make_gaussian_target(2, 'independent')
        p = MultivariateNormal(target.mean, target.covariance)
        self.assertAlmostEqual(result['metrics']['final_elbo'], -float(kl_divergence(q, p)))
        self.assertNotAlmostEqual(result['metrics']['final_elbo'], result['training_history'][0]['elbo'])
