"""Evaluator-only calculations. No calls to the target implementation or VDP."""
import math

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import logsumexp, ndtr, ndtri, roots_legendre

from .data import SE, Y

y, se = np.asarray(Y), np.asarray(SE)
L2PI = math.log(2*math.pi)
PROBS = np.array([.05,.5,.95])
NAMES = ['mu','tau',*[f'theta_{i}' for i in range(1,9)]]


def conditional(tau):
    t2 = np.atleast_1d(tau)[...,None]**2
    obsvar = se**2+t2
    w = 1/obsvar
    v = 1/(1/25+w.sum(axis=-1))
    m = v*(w*y).sum(axis=-1)
    loglike = -.5*(8*L2PI + np.log(obsvar).sum(axis=-1)+np.log(25/v)
                   + (w*(y-m[...,None])**2).sum(axis=-1)+m*m/25)
    shrink = t2/obsvar
    other = se**2/obsvar
    theta_m = shrink*y+other*m[...,None]
    theta_v = se**2*shrink + other**2*v[...,None]
    return loglike, np.column_stack([m,theta_m]), np.column_stack([v,theta_v])


def mixture_quantiles(cdf, mean, sd):
    values = []
    for prob in PROBS:
        width = 10*sd
        while cdf(mean-width)>prob or cdf(mean+width)<prob:
            width *= 2
        values.append(brentq(lambda x:cdf(x)-prob,mean-width,mean+width,xtol=2e-12))
    return values


def summarize(mean, sd, qs):
    return dict(mean=float(mean),sd=float(sd),q05=float(qs[0]),q50=float(qs[1]),q95=float(qs[2]))


def posterior(order):
    nodes, weights = roots_legendre(order)
    u = (nodes+1)/2
    tau = 5*np.tan(np.pi*u/2)
    ll, means, variances = conditional(tau)
    logs = np.log(weights/2)+ll
    logz = logsumexp(logs)
    p = np.exp(logs-logz)
    def tau_cdf(t):
        end = 2/np.pi*np.arctan(t/5)
        ls = conditional(5*np.tan(np.pi*u*end/2))[0]
        return end*np.dot(weights/2,np.exp(ls-logz))
    tau_q = [brentq(lambda t:tau_cdf(t)-prob,0,1000) for prob in PROBS]
    avg = p@tau
    stats = {'tau':summarize(avg,np.sqrt(p@(tau-avg)**2),tau_q)}
    for i,key in enumerate([NAMES[0],*NAMES[2:]]):
        loc, var = means[:,i],variances[:,i]
        m = p@loc
        sd = np.sqrt(p@(var+(loc-m)**2))
        qs=mixture_quantiles(lambda x:p@ndtr((x-loc)/np.sqrt(var)),m,sd)
        stats[key]=summarize(m,sd,qs)
    return dict(log_evidence=float(logz),marginals=stats)


def adaptive_evidence():
    # Independent path: integrate in tau with the dense y covariance directly.
    def f(t):
        cov = np.diag(se**2+t*t)+25*np.ones((8,8))
        ll = -.5*(8*L2PI+np.linalg.slogdet(cov)[1]+y@np.linalg.solve(cov,y))
        return np.exp(ll+32)*2/(5*np.pi*(1+(t/5)**2))
    mass, err = quad(f,0,np.inf,epsabs=1e-12,epsrel=1e-12)
    return np.log(mass)-32,err/mass


def normal_expect(fn):
    # Twelve standard deviations discard <4e-33 probability for bounded integrands.
    return quad(lambda e:np.exp(-.5*e*e)/np.sqrt(2*np.pi)*fn(e),-12,12,
                epsabs=2e-12,epsrel=2e-12)[0]


def fitted(form, end, truth):
    a = np.asarray(end['mu'])
    s = np.exp(end['effective_lambda'])
    v = s*s
    tau = np.exp(a[1]+v[1]/2)
    t2 = np.exp(2*a[1]+2*v[1])
    vt = np.expm1(v[1])*tau**2
    tau_q = np.exp(a[1]+s[1]*ndtri(PROBS))
    stats = {'mu':summarize(a[0],s[0],a[0]+s[0]*ndtri(PROBS)),
             'tau':summarize(tau,np.sqrt(vt),tau_q)}
    hyper = (-.5*(L2PI+np.log(25)+(a[0]**2+v[0])/25) + np.log(2/(5*np.pi))+a[1]
             - normal_expect(lambda e:np.logaddexp(0.,2*(a[1]+s[1]*e-np.log(5)))))
    if form=='centered':
        m, var = a[2:],v[2:]
        ez2 = np.exp(-2*a[1]+2*v[1])*((m-a[0])**2+var+v[0])
        prior = -.5*(ez2+L2PI).sum()-8*a[1]
    elif form=='noncentered':
        m = a[0]+tau*a[2:]
        var = v[0]+t2*v[2:]+vt*a[2:]**2
        prior = -.5*(a[2:]**2+v[2:]+L2PI).sum()
    else:
        raise ValueError(form)
    for j,key in enumerate(NAMES[2:]):
        sd = np.sqrt(var[j])
        if form=='centered':
            qs = m[j]+sd*ndtri(PROBS)
        else:
            def cdf(x):
                def conditional_cdf(e):
                    t = np.exp(a[1]+s[1]*e)
                    return ndtr((x-a[0]-t*a[j+2])/np.hypot(s[0],t*s[j+2]))
                return normal_expect(conditional_cdf)
            qs = mixture_quantiles(cdf,m[j],sd)
        stats[key]=summarize(m[j],sd,qs)
    like = -.5*np.sum(L2PI+2*np.log(se)+((y-m)**2+var)/se**2)
    entropy = .5*np.sum(1+L2PI+np.log(v))
    elbo = hyper+prior+like+entropy
    errors = {k:{stat:b-truth['marginals'][k][stat] for stat,b in vals.items()} for k,vals in stats.items()}
    return dict(reverse_kl=truth['log_evidence']-elbo,elbo=elbo,marginals=stats,errors=errors)
