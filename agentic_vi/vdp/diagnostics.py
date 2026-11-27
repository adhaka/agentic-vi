"""Descriptive estimates for deterministic optimizer iterates and iid proposal draws."""
import warnings

import arviz as az
import numpy as np
from scipy.special import logsumexp, ndtri
from scipy.stats import rankdata


def metric(value, units, n, status='ok', uncertainty=None, warnings=()):
    return dict(value=value, units=units, sample_count=n, status=status,
                estimator_version='vdp/0.1', uncertainty=uncertainty, warnings=list(warnings))


def classical(x):
    within = x.var(axis=1, ddof=1).mean(axis=0)
    between = x.mean(axis=1).var(axis=0, ddof=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.sqrt((x.shape[1]-1)/x.shape[1] + between/within)


def ranks(x):
    flat = x.reshape(-1, x.shape[-1])
    r = rankdata(flat, axis=0)
    return ndtri((r-.375)/(len(flat)+.25)).reshape(x.shape)


def rhat(x):
    """Input contains four aligned, already-windowed post-update series."""
    if x is None or len(x) != 4:
        return metric(None, 'ratio', 0, 'run_failed')
    n = x.shape[1]
    if n < 40:
        return metric(None, 'ratio', n, 'insufficient_samples')
    if not np.isfinite(x).all():
        return metric(None, 'ratio', n, 'nonfinite')
    split = np.concatenate((x[:, :n//2], x[:, -(n//2):]))
    rank = classical(ranks(split))
    folded = classical(ranks(abs(split - np.median(split, axis=(0, 1)))))
    records = {}
    for name, vals in [('classical', classical(x)), ('split', classical(split)),
                       ('rank', rank), ('folded', folded), ('maximum', np.maximum(rank, folded))]:
        statuses = np.where(np.isinf(vals), 'separated_constant', np.where(np.isnan(vals), 'undefined', 'ok'))
        if name == 'maximum':
            statuses[np.isinf(rank) | np.isinf(folded)] = 'separated_constant'
        good = np.isfinite(vals)
        records[name] = dict(values=[float(v) if np.isfinite(v) else None for v in vals],
                             coordinate_status=statuses.tolist(), quantile_levels=[0,.5,.9,.95,1],
                             quantiles=np.quantile(vals,[0,.5,.9,.95,1]) if good.all() else None,
                             fraction_above_1_01=np.mean(vals>1.01) if good.all() else None)
    return metric(records, 'ratio', 4*n, 'degenerate' if not good.all() else 'ok')


def drift(window):
    half = len(window)//2
    if half == 0:
        return metric(None, 'standardized', len(window), 'insufficient_samples')
    early, late = window[:half].mean(axis=0), window[-half:].mean(axis=0)
    mu, lam = np.split(late-early, 2)
    mu /= np.exp(np.split((early+late)/2, 2)[1])
    return metric({name: dict(signed=a, rms=np.sqrt(np.mean(a*a)), max_abs=max(abs(a)))
                   for name,a in [('mu',mu),('lambda',lam)]}, 'standardized', 2*half)


def gradient(g, eps, sigma):
    h = np.concatenate((g*sigma, 1+g*sigma*eps), axis=1)
    avg, se = h.mean(axis=0), h.std(axis=0, ddof=1)/np.sqrt(len(h))
    return metric(dict(mean=avg, rms=np.sqrt(np.mean(avg**2)), max_abs=max(abs(avg))),
                  'standardized gradient', len(g), uncertainty=dict(se=se, noise_rms=np.sqrt(np.mean(se**2))))


def agreement(a, b):
    mu, la = a
    mv, lb = b
    diff = mu-mv
    dl = la-lb
    mean = .25*np.sum(diff**2*(np.exp(-2*la)+np.exp(-2*lb)))
    scale = np.sum(np.sinh(dl)**2)
    return dict(joint=mean+scale, per_dimension=(mean+scale)/len(mu), mean=mean, scale=scale,
                standardized_delta=diff/np.sqrt((np.exp(2*la)+np.exp(2*lb))/2), delta_lambda=dl)


def importance(a):
    n = len(a)
    zero = np.isneginf(a)
    if np.isnan(a).any() or np.isposinf(a).any() or zero.all():
        return metric(dict(zero_weights=int(zero.sum())), 'importance diagnostics', n, 'nonfinite')
    centered = a - a.max()
    w = np.exp(centered-logsumexp(centered))
    ess = 1/(w@w)
    notes = []
    if zero.any():
        notes.append('zero target density, ELBO is negative infinity')
    result = dict(raw_ess=ess, relative_ess=ess/n, max_weight=w.max(), zero_weights=int(zero.sum()),
                  elbo=None if zero.any() else a.mean(), k_ref=min(1-1/np.log10(n),.7),
                  k_hat=None, k_interval=None, psis_ess=None, smoothed_log_weights=None,
                  library='arviz/'+az.__version__, reff=1)
    se = None if zero.any() else a.std(ddof=1)/np.sqrt(n)
    if np.all(centered==0):
        result.update(raw_ess=n, relative_ess=1., psis_ess=n, smoothed_log_weights=np.full(n,-np.log(n)))
        return metric(result, 'weights and nats', n, 'degenerate', dict(elbo_se=se), ['constant log weights'])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lw, k = az.psislw(centered, reff=1.)
    notes.extend(str(w.message) for w in caught)
    result.update(smoothed_log_weights=lw, psis_ess=1/np.exp(2*lw).sum())
    if np.isfinite(k):
        result['k_hat'] = float(k)
        if k > result['k_ref']:
            notes.append('Pareto shape exceeds sample-size reference')
    else:
        notes.append('undefined Pareto fit')
    return metric(result, 'weights and nats', n, 'inconclusive' if zero.any() or not np.isfinite(k) else 'ok',
                  dict(elbo_se=se), notes)


def energies(u, z):
    f = (u**2).sum(axis=2).mean(axis=1)
    t = u*z
    off = t.var(axis=1, ddof=1).sum(axis=1)
    variation = u.var(axis=0, ddof=1).sum(axis=1).mean()
    return f, off, variation


def curvature(u, z, rng, repetitions=200):
    j,r,d = u.shape
    if min(j,r)<2 or not np.isfinite(u).all():
        return metric(None,'ratio',j*r,'insufficient_samples' if min(j,r)<2 else 'nonfinite')
    f, off, v = energies(u,z)
    if f.mean()==0:
        return metric(dict(total_energy=f,off_energy=off,variation=v), 'ratio',j*r,'degenerate')
    dep, geom = np.sqrt([off.mean()/f.mean(), v/f.mean()])
    samples = []
    for _ in range(repetitions):
        ji = rng.integers(j,size=j)
        ri = rng.integers(r,size=r)
        bf,bo,bv = energies(u[ji[:,None],ri], z[ri])
        if bf.mean()>0:
            samples.append(np.sqrt([bo.mean()/bf.mean(), bv/bf.mean()]))
    directional = (u*z).mean(axis=2)
    local = [float(np.sqrt(o/k)) if k else None for o,k in zip(off,f)]
    notes = []
    if dep>1 or any(i is not None and i>1 for i in local) or geom>1:
        notes.append('finite-probe ratio above one')
    return metric(dict(dependence=dep, geometry=geom, total_energy=f, off_energy=off,
                       local_ratios=local, variation=v, energy_per_dimension=f.mean()/d,
                       estimated_diagonals=(u*z).mean(axis=1), directional=directional,
                       min_directional=directional.min(), negative_fraction=np.mean(directional<0)),
                  'standardized curvature',j*r,uncertainty=dict(replicates=repetitions,
                  invalid=repetitions-len(samples),percentiles=np.percentile(samples,[2.5,97.5],axis=0)
                  if samples else None),warnings=notes)
