from typing import Any, Callable, Dict, Sequence

import torch
from tqdm.auto import tqdm

from banditeval.bandits import (
    upper_confidence_bound_exploration,
    upper_confidence_bound_exploration_low_rank_factorization,
)


class BanditEval:
    def __init__(
        self,
        methods: Sequence[Callable],
        dataset: Any,
        evaluation_function: Callable,
        algorithm: str,
        observation_matrix: torch.Tensor = None,
    ) -> None:
        if algorithm == "ucbe":
            self.step = upper_confidence_bound_exploration
        elif algorithm == "ucbe-lrf":
            self.step = upper_confidence_bound_exploration_low_rank_factorization
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}.")
        self.observation_matrix = (
            torch.full((len(methods), len(dataset)), torch.nan)
            if observation_matrix is None
            else observation_matrix
        )

        self.methods = methods
        self.dataset = dataset
        self.evaluation_function = evaluation_function
        self.algorithm = algorithm

    def __call__(self, budget: int, **kwargs: Dict[Any, str]) -> torch.Tensor:
        for _ in tqdm(range(budget)):
            indices = self.step(self.observation_matrix, return_mus=False, **kwargs)
            if indices is None:
                break
            else:
                row_indices, column_indices = indices
            i = row_indices[0]
            method = self.methods[i]
            examples = [self.dataset[j.item()] for j in column_indices]
            scores = self.evaluation_function(method, examples)
            self.observation_matrix[row_indices, column_indices] = scores
        _, method_scores = self.step(self.observation_matrix, return_mus=True, **kwargs)
        return method_scores
