import warnings

import torch

from banditeval.factorization import Factorization


def upper_confidence_bound_exploration(
    observed_matrix: torch.Tensor,
    a: int = 1,
    batch_size: int = 32,
    return_mus: bool = False,
):
    """Single step of Upper Confidence Bound Exploration"""
    m_methods, n_examples = observed_matrix.shape
    # Update bounds
    bounds = torch.full((m_methods,), torch.inf)
    mus = observed_matrix.nanmean(dim=1)
    counts = (~observed_matrix.isnan()).sum(1)
    mask = counts > 0
    bounds[mask] = mus[mask] + torch.sqrt(a / counts[mask])
    completely_sensed_mask = counts == n_examples
    bounds[completely_sensed_mask] = -torch.inf
    # Select batch
    if completely_sensed_mask.sum() == m_methods:
        batch = None
    else:
        best_method_index = torch.argmax(bounds)
        unobserved_column_indices = (
            observed_matrix[best_method_index].isnan().nonzero().flatten()
        )
        n_unobserved = unobserved_column_indices.size(0)
        batch_size = min(batch_size, n_unobserved)
        batch = torch.stack(
            [
                torch.full((batch_size,), best_method_index),
                unobserved_column_indices[torch.randperm(n_unobserved)[:batch_size]],
            ]
        )
    if return_mus:
        return batch, mus
    return batch


def upper_confidence_bound_exploration_low_rank_factorization(
    observed_matrix: torch.Tensor,
    a: int = 1,
    batch_size: int = 32,
    rank: int = 1,
    ensemble_size: int = 64,
    warmup_percentage: float = 0.05,
    regularizer_weight: float = 0.1,
    drop_probability: float = 0.05,
    iterations: int = 10,
    eta: float = 5,
    device: str = "cpu",
    return_mus: bool = False,
):
    """Single step of Upper Confidence Bound Exploration with Low Rank Factorization"""
    if warmup_percentage < 0.05:
        warnings.warn(
            "Low warmup percentage can significantly degrade performance.", UserWarning
        )
    observed_matrix = observed_matrix.to(device)
    if (1 - observed_matrix.isnan().sum() / observed_matrix.numel()) < warmup_percentage:
        row_indices, column_indices = torch.where(observed_matrix.isnan())
        sample_indices = torch.randperm(row_indices.size(0))[:batch_size]
        batch = torch.stack([row_indices[sample_indices], column_indices[sample_indices]])
        if return_mus:
            warnings.warn("During warmup period scores are not computed.", UserWarning)
            return batch, None
        return batch
    m_methods, n_examples = observed_matrix.shape
    # Update bounds
    bounds = torch.full((m_methods,), torch.inf, device=device)
    factorization = Factorization(
        m_methods,
        n_examples,
        rank,
        ensemble_size,
        regularizer_weight=regularizer_weight,
        drop_probability=drop_probability,
    ).to(device)
    factorization.fit(observed_matrix, iterations=iterations)
    matrix_approximation = factorization()
    entry_mus = matrix_approximation.mean(0)
    entry_stds = matrix_approximation.std(0)
    entry_ucb = entry_mus + eta * entry_stds
    observed_mask = ~observed_matrix.isnan()
    entry_ucb[observed_mask] = observed_matrix[observed_mask]
    ucb = entry_ucb.mean(1)
    counts = observed_mask.sum(1)
    mask = counts > 0
    bounds[mask] = ucb[mask]
    completely_sensed_mask = counts == n_examples
    bounds[completely_sensed_mask] = -torch.inf
    # Select batch
    if completely_sensed_mask.sum() == m_methods:
        batch = None
    else:
        best_method_index = torch.argmax(bounds)
        best_method_stds = entry_stds[best_method_index]
        best_method_stds[observed_mask[best_method_index]] = -1
        batch_size = min(batch_size, torch.sum(~observed_mask[best_method_index]))
        batch = torch.stack(
            [
                torch.full((batch_size,), best_method_index, device=device),
                torch.topk(best_method_stds, batch_size, largest=True).indices,
            ]
        ).cpu()
    if return_mus:
        entry_mus[observed_mask] = observed_matrix[observed_mask]
        mus = entry_mus.mean(1).cpu()
        return batch, mus
    return batch
