"""Synthetic multivariate Gaussian targets with known ground truth."""
from dataclasses import dataclass
import torch


@dataclass
class GaussianTarget:
    mean: torch.Tensor
    covariance: torch.Tensor

    def __post_init__(self):
        self.precision = torch.linalg.inv(self.covariance)
        self.logdet_covariance = torch.linalg.slogdet(self.covariance).logabsdet
        self.dimension = self.mean.numel()
        self._normalizer = (
            self.dimension * torch.log(torch.tensor(2.0 * torch.pi, dtype=self.mean.dtype))
            + self.logdet_covariance
        )

    def log_prob(self, theta):
        delta = theta - self.mean
        quadratic = torch.einsum("...i,ij,...j->...", delta, self.precision, delta)
        return -0.5 * (self._normalizer + quadratic)


def make_gaussian_target(dimension: int, kind: str = "correlated",
                         dtype=torch.float64, device="cpu") -> GaussianTarget:
    """Create a deterministic target; randomness is not used in its construction."""
    mean = torch.linspace(-0.5, 0.5, dimension, dtype=dtype, device=device)

    if kind == "independent":
        variances = torch.linspace(0.5, 2.0, dimension, dtype=dtype, device=device)
        covariance = torch.diag(variances)
    elif kind == "correlated":
        # Positive-definite Toeplitz covariance with moderate correlation.
        idx = torch.arange(dimension, dtype=dtype, device=device)
        distance = (idx[:, None] - idx[None, :]).abs()
        covariance = 0.75 ** distance
        scale = torch.linspace(0.8, 1.3, dimension, dtype=dtype, device=device)
        covariance = scale[:, None] * covariance * scale[None, :]
    elif kind == "ill_conditioned":
        # Orthogonal rotation with a controlled eigenvalue range.
        raw = torch.arange(1, dimension * dimension + 1, dtype=dtype, device=device)
        matrix = torch.sin(raw.reshape(dimension, dimension))
        orthogonal, _ = torch.linalg.qr(matrix)
        eigenvalues = torch.logspace(0, -3, dimension, dtype=dtype, device=device)
        covariance = orthogonal @ torch.diag(eigenvalues) @ orthogonal.T
        covariance = (covariance + covariance.T) / 2
    else:
        raise ValueError(f"Unknown target kind: {kind}")

    return GaussianTarget(mean=mean, covariance=covariance)
