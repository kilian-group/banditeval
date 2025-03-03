import pytest
import torch

from banditeval.bandits import (
    upper_confidence_bound_exploration,
    upper_confidence_bound_exploration_low_rank_factorization,
)


@pytest.fixture
def create_obvious_winner_matrix():
    """ """

    def _create_matrix(m_methods, n_examples):
        X = torch.full((m_methods, n_examples), torch.nan)
        indices = torch.randperm(m_methods)
        winner, losers = indices[0:1], indices[1:]
        X[winner, : n_examples // 2] = 1
        X[losers, : n_examples // 2] = 0
        return X, winner.item()

    return _create_matrix


@pytest.fixture
def create_nearly_full_matrix():
    """ """

    def _create_matrix(m_methods, n_examples):
        X = torch.rand(m_methods, n_examples).round()
        method_index = torch.randint(m_methods, (1,))
        example_index = torch.randint(n_examples, (1,))
        X[method_index, example_index] = torch.nan
        return X, method_index.item()

    return _create_matrix


def test_upper_confidence_bound_exploration_obvious_winner(
    create_obvious_winner_matrix,
):
    """ """
    X, i = create_obvious_winner_matrix(10, 10)
    batch = upper_confidence_bound_exploration(X, a=1, batch_size=1)
    assert batch[0][0].item() == i


def test_upper_confidence_bound_exploration_nearly_full(create_nearly_full_matrix):
    """ """
    X, i = create_nearly_full_matrix(10, 10)
    batch = upper_confidence_bound_exploration(X, a=1, batch_size=1)
    assert batch[0][0].item() == i


def test_upper_confidence_bound_exploration_low_rank_factorization_obvious_winner(
    create_obvious_winner_matrix,
):
    """ """
    X, i = create_obvious_winner_matrix(10, 10)
    batch = upper_confidence_bound_exploration_low_rank_factorization(
        X, a=1, batch_size=1
    )
    assert batch[0][0].item() == i


def test_upper_confidence_bound_exploration_low_rank_factorization_nearly_full(
    create_nearly_full_matrix,
):
    """ """
    X, i = create_nearly_full_matrix(10, 10)
    batch = upper_confidence_bound_exploration_low_rank_factorization(
        X, a=1, batch_size=1
    )
    assert batch[0][0].item() == i
