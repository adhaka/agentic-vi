"""Mean-field Gaussian variational inference baseline."""
from dataclasses import dataclass
import time
import torch
from torch import nn
from torch.distributions import Independent, Normal


@dataclass
class MFVIConfig:
    dimension: int
    steps: int = 3000
    learning_rate: float = 0.03
    mc_samples: int = 128
    log_every: int = 100
    dtype: torch.dtype = torch.float64
    device: str = "cpu"


class MeanFieldGaussianVI(nn.Module):
    """q(theta) = product_i Normal(theta_i; mu_i, sigma_i^2)."""

    def __init__(self, config: MFVIConfig):
        super().__init__()
        self.config = config
        self.mu = nn.Parameter(torch.zeros(
            config.dimension, dtype=config.dtype, device=config.device
        ))
        self.log_std = nn.Parameter(torch.zeros(
            config.dimension, dtype=config.dtype, device=config.device
        ))

    def distribution(self):
        # Clamp only for numerical safety; this is part of the baseline algorithm.
        std = self.log_std.clamp(min=-12.0, max=8.0).exp()
        return Independent(Normal(self.mu, std), 1)

    def fit(self, target):
        """Maximize a Monte Carlo ELBO for a fixed number of optimizer steps."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        history = []
        start = time.perf_counter()

        for step in range(1, self.config.steps + 1):
            q = self.distribution()
            theta = q.rsample((self.config.mc_samples,))
            log_p = target.log_prob(theta)
            log_q = q.log_prob(theta)
            elbo = (log_p - log_q).mean()

            optimizer.zero_grad(set_to_none=True)
            (-elbo).backward()
            optimizer.step()

            if step == 1 or step % self.config.log_every == 0 or step == self.config.steps:
                history.append({"step": step, "elbo": float(elbo.detach().cpu())})

        runtime = time.perf_counter() - start
        return {"history": history, "runtime_seconds": runtime}

    @torch.no_grad()
    def moments(self):
        std = self.log_std.clamp(min=-12.0, max=8.0).exp()
        return self.mu.detach().clone(), std.square().detach().clone()
