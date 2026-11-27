"""Joint log densities with respect to the two unconstrained coordinate systems."""
import math

import torch
from .data import SE, Y

LOG2PI = math.log(2 * math.pi)


def density(form):
    if form not in ('centered', 'noncentered'):
        raise ValueError(form)
    y = torch.tensor(Y, dtype=torch.float64)
    se = torch.tensor(SE, dtype=torch.float64)

    def log_joint(x):
        mu, eta, w = x[..., 0], x[..., 1], x[..., 2:]
        # Half-Cauchy density times d tau/d eta, in log space.
        scale = math.log(2/(5*math.pi)) + eta - torch.logaddexp(
            torch.zeros_like(eta), 2*(eta-math.log(5)))
        hyper = scale - .5*(mu/5)**2 - math.log(5) - .5*LOG2PI
        if form == 'centered':
            theta = w
            z = (theta-mu[..., None])*torch.exp(-eta[..., None])
            log_scale = -len(Y)*eta
        else:
            z = w
            theta = mu[..., None]+torch.exp(eta[..., None])*z
            log_scale = 0.
        prior = -.5*(z.square()+LOG2PI).sum(dim=-1)+log_scale
        likelihood = (-.5*(((y-theta)/se).square()+LOG2PI)-se.log()).sum(dim=-1)
        return hyper+prior+likelihood
    return log_joint
