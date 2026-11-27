"""Check archived fits using Gaussian cubature and the saved target matrices."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


def audit(directory):
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert manifest['failures'] == []
    expected = set()
    rows = []
    for kind, d in itertools.product(('independent', 'correlated', 'ill_conditioned'), (10, 50, 100)):
        cov = np.load(directory / f'{kind}_d{d}_cov.npy', allow_pickle=False)
        mean = np.linspace(-.5, .5, d)
        chol = np.linalg.cholesky(cov)
        # P = L^-T L^-1. The squared column norms give its diagonal.
        opt = 1 / np.square(np.linalg.solve(chol, np.eye(d))).sum(axis=0)
        posterior = np.diag(cov)

        def cubature_kl(mu, var):
            # Symmetric points reproduce the first two moments of q exactly.
            # Gaussian log densities are quadratic so this rule integrates them exactly.
            shift = np.diag(np.sqrt(d * var))
            points = np.concatenate([mu + shift, mu - shift])
            white = np.linalg.solve(chol, (points - mean).T)
            return np.log(np.diag(chol)).sum() - np.log(var).sum()/2 + (np.square(white).sum(axis=0).mean()-d)/2

        floor = cubature_kl(mean, opt)
        for seed in range(5):
            path = directory / f'{kind}_d{d}_seed{seed}.json'
            expected.add(path)
            r = json.loads(path.read_text())
            assert (r['dimension'], r['target'], r['seed']) == (d, kind, seed)
            assert r['evaluation_schema'] == 2
            assert r['config'] == dict(steps=3000, mc_samples=128, learning_rate=.03,
                                       dtype='float64', log_every=100)
            assert r['covariance_sha256'] == hashlib.sha256(cov.tobytes()).hexdigest()
            history = r['training_history']
            assert [h['step'] for h in history] == [1, *range(100, 3001, 100)]
            assert all(np.isfinite(h['elbo']) for h in history)
            mu, var = map(np.asarray, (r['fitted_mean'], r['fitted_variance']))
            assert mu.shape == var.shape == (d,) and (var > 0).all()
            total = cubature_kl(mu, var)
            ref = dict(exact_reverse_kl_q_to_p=total, irreducible_mean_field_kl=floor,
                       optimization_gap_kl=total-floor, final_elbo=-total,
                       posterior_mean_mse=np.mean((mu-mean)**2),
                       optimal_mean_field_variance_mse=np.mean((var-opt)**2),
                       posterior_marginal_variance_mse=np.mean((var-posterior)**2))
            metrics = r['metrics']
            assert metrics['all_parameters_finite']
            for key, value in ref.items():
                assert np.isfinite(value)
                np.testing.assert_allclose(metrics[key], value, rtol=1e-9, atol=1e-10,
                                           err_msg=f'{path.name}: {key}')
            rows.append(dict(target=kind, d=d, seed=seed, **ref,
                             min_variance_ratio=np.min(var/posterior),
                             max_variance_ratio=np.max(var/posterior),
                             runtime=r['runtime_seconds']))
    assert expected == set(directory.glob('*_seed*.json'))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = audit(args.directory)
    print(f'Validated {len(rows)} run reports.')
