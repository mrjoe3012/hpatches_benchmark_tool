from __future__ import annotations
from dataclasses import dataclass
from ndshapecheck import ShapeCheck
import numpy as np
from hpatches_benchmark.benchmark.matches import Matches
from hpatches_benchmark.benchmark.tabular import Tabular

@dataclass
class MMAEvaluation(Tabular):
    """
    Mean matching accuracy evaluation. Computed as the number of matched pairs within a distance
        threshold when transformed by the ground truth homography.
    :param matches:
    :param epsilon:
    :param mma: One per epsilon value.
    """
    matches: Matches
    epsilon: np.ndarray[np.float64, 1]
    mma: np.ndarray[np.float64, 1]

    def __post_init__(self) -> None:
        sc = ShapeCheck()
        assert sc('N').check(self.epsilon), sc.why
        assert sc('N').check(self.mma), sc.why

    @property
    def table_headings(self):
        return [
            f'MMA @ {eps:.2f}' for eps in self.epsilon
        ]

    @property
    def table_body(self):
        return self.mma.tolist()

    @staticmethod
    def construct_empty(matches: Matches, epsilon: np.ndarray[np.float64, 1]) -> MMAEvaluation:
        return MMAEvaluation(
            matches,
            epsilon,
            np.full(epsilon.shape, 0.0)
        )
 