from __future__ import annotations
from dataclasses import dataclass
from ndshapecheck import ShapeCheck
import numpy as np

from hpatches_benchmark.benchmark.matches import Matches
from hpatches_benchmark.benchmark.tabular import Tabular

__all__ = ['MatchingScoreEvaluation']

@dataclass
class MatchingScoreEvaluation(Tabular):
    """
    Evaluation of the matching score. Computed as the number of matched shared viewpoint keypoints
        which were within some threshold after being transformed by the ground truth homography.
        Computed symmetrically for each image pair and averaged.
    :param matches:
    :param epsilon:
    :param m_score: One per epsilon value.
    """
    matches: Matches
    epsilon: np.ndarray[np.float64, 1]
    m_score: np.ndarray[np.float64, 1]

    def __post_init__(self) -> None:
        sc = ShapeCheck()
        assert sc('N').check(self.epsilon), sc.why
        assert sc('N').check(self.m_score), sc.why

    @property
    def table_headings(self):
        return [
            f'M SCORE @ {eps:.2f}' for eps in self.epsilon
        ]

    @property
    def table_body(self):
        return self.m_score.tolist()
    
    @staticmethod
    def construct_empty(matches: Matches, epsilon: np.ndarray) -> MatchingScoreEvaluation:
        return MatchingScoreEvaluation(
            matches, epsilon, np.full(epsilon.shape, 0.0)
        )
