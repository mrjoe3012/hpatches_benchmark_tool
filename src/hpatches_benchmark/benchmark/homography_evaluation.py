from dataclasses import dataclass
from ndshapecheck import ShapeCheck
import numpy as np

from hpatches_benchmark.benchmark.homography_estimate import HomographyEstimate
from hpatches_benchmark.benchmark.tabular import Tabular

__all__ = ['HomographyEvaluation']

@dataclass
class HomographyEvaluation(Tabular):
    """
    Evaluation of estimated homography. 
    :param homography_estimate:
    :param epsilon: The thresholds used to calculate the metrics.
    :param mean_localisation_err: The MLE for each of the thresholds in epsilon.
    :param corrrect_homographies: Proportion of correct homographies for each threshold in
        epsilon.
    :param mean_corner_error: Mean L2 norm error of the projected image corners.
    """
    homography_estimate: HomographyEstimate
    epsilon: np.ndarray[np.float64, 1]
    mean_localisation_err: np.ndarray[np.float64, 1]
    correct_homographies: np.ndarray[np.float64, 1]
    mean_corner_error: float

    def __post_init__(self) -> None:
        sc = ShapeCheck()
        assert sc('N').check(self.epsilon), sc.why
        assert sc('N').check(self.mean_localisation_err), sc.why
        assert sc('N').check(self.correct_homographies), sc.why

    @property
    def table_headings(self):
        return [
            f'Correct Homo @ {eps:.2f}' for eps in self.epsilon
        ] + [
            f'MLE @ {eps:.2f}' for eps in self.epsilon
        ] + ['Num Points']

    @property
    def table_body(self):
        return self.correct_homographies.tolist() + self.mean_localisation_err.tolist() \
            + [self.homography_estimate.kp_prediction.shape[0]]
