"""Independent evaluation utilities. Do not expose this module for agent edits."""
import torch


@torch.no_grad()
def exact_kl_diagonal_gaussian_to_target(q_mean, q_variance, target):
    """Exact KL[N(q_mean, diag(q_variance)) || N(target.mean, target.covariance)]."""
    d = q_mean.numel()
    delta = q_mean - target.mean
    trace_term = torch.sum(torch.diagonal(target.precision) * q_variance)
    quadratic_term = delta @ target.precision @ delta
    logdet_q = torch.log(q_variance).sum()
    return 0.5 * (
        trace_term + quadratic_term - d
        + target.logdet_covariance - logdet_q
    )


@torch.no_grad()
def evaluate_mfvi(model, target, final_elbo=None):
    q_mean, q_variance = model.moments()
    target_mean = target.mean
    # The reverse-KL-optimal diagonal Gaussian has variance 1 / diag(precision).
    optimal_mf_variance = 1.0 / torch.diagonal(target.precision)

    metrics = {
        "exact_reverse_kl_q_to_p": float(
            exact_kl_diagonal_gaussian_to_target(q_mean, q_variance, target).cpu()
        ),
        "mean_squared_error": float(torch.mean((q_mean - target_mean) ** 2).cpu()),
        "diagonal_variance_mse": float(
            torch.mean((q_variance - optimal_mf_variance) ** 2).cpu()
        ),
        "all_parameters_finite": bool(
            torch.isfinite(q_mean).all() and torch.isfinite(q_variance).all()
        ),
    }
    if final_elbo is not None:
        metrics["final_elbo"] = float(final_elbo)
    return metrics
