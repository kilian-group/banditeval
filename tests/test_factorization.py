import pytest
import torch

from banditeval.factorization import Factorization


@pytest.fixture
def create_matrix():
    """ """

    def _create_matrix(m_methods, n_examples, rank):
        U = torch.rand(m_methods, rank)
        V = torch.rand(n_examples, rank)
        return torch.mm(U, V.T), U, V

    return _create_matrix


def test_single_factorization(create_matrix):
    """ """
    m_methods, n_examples = 10, 10
    ranks = [1, 3, 5]
    for rank in ranks:
        X, U, V = create_matrix(m_methods, n_examples, rank)
        factorization = Factorization(m_methods, n_examples, rank, 1, 0.1, 0.0)
        factorization.fit(X)
        assert torch.allclose(factorization().squeeze(0), X, atol=rank)
        assert torch.allclose(factorization.U.squeeze(0).abs(), U.abs(), atol=rank)
        assert torch.allclose(factorization.V.squeeze(0).abs(), V.abs(), atol=rank)


def test_ensembled_factorization(create_matrix):
    """ """
    m_methods, n_examples = 10, 10
    ranks = [1, 3, 5]
    for rank in ranks:
        X, U, V = create_matrix(m_methods, n_examples, rank)
        factorization = Factorization(m_methods, n_examples, rank, 32, 0.1, 0.05)
        factorization.fit(X)
        assert torch.allclose(factorization().mean(0), X, atol=rank)
        assert torch.allclose(factorization.U.mean(0).abs(), U.abs(), atol=rank)
        assert torch.allclose(factorization.V.mean(0).abs(), V.abs(), atol=rank)
