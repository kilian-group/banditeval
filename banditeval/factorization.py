import torch
from einops import rearrange


class Factorization(torch.nn.Module):
    r"""
    Factorization model of the form $$X = UV^T$$

    where $X$ is the matrix to be factorized, $U$ and $V$ are the factor matrices.
    """

    def __init__(
        self,
        m_methods: int,
        n_examples: int,
        rank: int,
        ensemble_size: int,
        regularizer_weight: float = 0.00,
        drop_probability: float = 0.05,
    ) -> None:
        """
        Initialize the factorization model.

        Args:
            m_methods: Number of methods (i.e. rows of the matrix)
            n_examples: Number of examples (i.e. columns of the matrix)
            rank: Rank of the factorization
            ensemble_size: Number of factorization models in the ensemble
            regularizer_weight: Weight of the L2 regularizer
            drop_probability: Probability of dropping an entry in the matrix
        """
        super().__init__()
        self.register_buffer("U", torch.randn(ensemble_size, m_methods, rank))
        self.register_buffer("V", torch.randn(ensemble_size, n_examples, rank))
        self.register_buffer("L", regularizer_weight * torch.eye(rank))

        self.m_methods = m_methods
        self.n_examples = n_examples
        self.rank = rank
        self.ensemble_size = ensemble_size
        self.regularizer_weight = regularizer_weight
        self.drop_probability = drop_probability

    def forward(self) -> torch.Tensor:
        """
        Compute the approximation of the matrix using the factor matrices.
        """
        return torch.bmm(self.U, self.V.transpose(1, 2))

    def _als_step(
        self, data_matrix: torch.Tensor, fixed_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform a single alternating least squares step.
        """
        non_zero_mask = (~torch.isnan(data_matrix)).float()
        y = fixed_matrix.unsqueeze(2)
        y_t = y.transpose(1, 2)
        A = (non_zero_mask.unsqueeze(2) * torch.bmm(y, y_t)).sum(0) + self.L
        b = (torch.nan_to_num(data_matrix * non_zero_mask) * y.squeeze(2)).sum(0)
        return torch.linalg.solve(A, b)

    def fit(self, X: torch.Tensor, iterations: int = 10) -> None:
        r"""
        Solve the following using alternating least squares

        $$\min_{U, V} \Big|\Big|O\odot\Big(UV^T - X\Big)\Big|\Big|_F^2$$

        where $O$ is the binary observation matrix (O_{ij} = 1 if X_{ij} is not
        nan else 0), $\odot$ is the Hadamard product, and $||\cdot||_F$ is the
        Frobenius norm.
        """
        X = X.unsqueeze(0).repeat(self.ensemble_size, 1, 1)
        if self.drop_probability > 0:
            mask = torch.rand_like(X) < self.drop_probability
            X[mask] = torch.nan
        X_u = rearrange(X, "e m n -> (e m) n 1")
        X_v = rearrange(X, "e m n -> (e n) m 1")
        vmap_als_step = torch.vmap(self._als_step, in_dims=(0, 0))
        for _ in range(iterations):
            self.V.data = (
                vmap_als_step(X_v, self.U.repeat(self.n_examples, 1, 1))
            ).reshape(self.ensemble_size, self.n_examples, self.rank)
            self.U.data = (
                vmap_als_step(X_u, self.V.repeat(self.m_methods, 1, 1))
            ).reshape(self.ensemble_size, self.m_methods, self.rank)

    def reset(self) -> None:
        """
        Reset the factor matrices to random values.
        """
        torch.nn.init.normal_(self.U)
        torch.nn.init.normal_(self.V)
