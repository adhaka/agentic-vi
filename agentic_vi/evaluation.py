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
def evaluate_mfvi(model, target):
    """Evaluate the final distribution against the normalized Gaussian target."""
    mean, variance = model.moments()
    precision_diag = torch.diagonal(target.precision)
    optimal_variance = precision_diag.reciprocal()
    delta = mean - target.mean
    total = exact_kl_diagonal_gaussian_to_target(mean, variance, target)
    floor = 0.5 * (target.logdet_covariance + precision_diag.log().sum())
    # Compute excess KL directly to avoid subtracting two large total KLs.
    log_ratio = variance.log() + precision_diag.log()
    gap = 0.5 * (delta @ target.precision @ delta
                 + (log_ratio.expm1() - log_ratio).sum())
    mean_mse = float(delta.square().mean())
    optimal_variance_mse = float((variance - optimal_variance).square().mean())
    return {
        "exact_reverse_kl_q_to_p": float(total),
        "irreducible_mean_field_kl": float(floor),
        "optimization_gap_kl": float(gap),
        "posterior_mean_mse": mean_mse,
        "optimal_mean_field_variance_mse": optimal_variance_mse,
        "posterior_marginal_variance_mse": float(
            (variance - target.covariance.diagonal()).square().mean()
        ),
        "final_elbo": -float(total),
        "all_parameters_finite": all(bool(p.isfinite().all()) for p in model.parameters()),
        # Legacy v0 names retained for existing report readers.
        "mean_squared_error": mean_mse,
        "diagonal_variance_mse": optimal_variance_mse,
    }
