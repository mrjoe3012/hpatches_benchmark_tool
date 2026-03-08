from __future__ import annotations
from dataclasses import dataclass

from ndshapecheck import ShapeCheck
import numpy as np

from hpatches_benchmark.benchmark.homography_estimate import HomographyEstimate
from hpatches_benchmark.benchmark.matches import Matches

__all__ = ['HomographyEvaluationIOU']

@dataclass
class HomographyEvaluationIOU:
    """
    Evaluate the correctness of a homography via the intersection over union of the image bounding
        bounding box after being transformed by the predicted and ground truth homographies. The
        image bounding boxes are computed using the standard image size.
    :param homography_estimate: The estimated homography.
    :param epsilon: Epsilon values at which to consider the intersection over union is
        considered correct.
    """
    homography_estimate: HomographyEstimate
    epsilon: np.ndarray[np.float64, 1]
    correct_homographies: np.ndarray[np.float64, 1]
    intersection_over_union: float

    def __post_init__(self) -> None:
        sc = ShapeCheck()
        assert sc('N').check(self.epsilon)
        assert sc('N').check(self.correct_homographies)

    @property
    def table_headings(self):
        return [f'IOU @ {eps}' for eps in self.epsilon]

    @property
    def table_body(self):
        return self.correct_homographies.tolist()

    @staticmethod
    def construct_empty(matches: Matches,
                        epsilon: np.ndarray[np.float64, 1]) -> HomographyEvaluationIOU:
        return HomographyEvaluationIOU(
            HomographyEstimate.construct_empty(matches),
            epsilon,
            np.full(epsilon.shape, 0),
            0.0,
        )
