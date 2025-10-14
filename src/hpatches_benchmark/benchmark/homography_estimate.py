from __future__ import annotations
from dataclasses import dataclass

from ndshapecheck import ShapeCheck
import numpy as np

from hpatches_benchmark.benchmark.matches import Matches

__all__ = ['HomographyEstimate']

@dataclass
class HomographyEstimate:
    """
    Homography estimated from feature matches.
    :param matches:
    :param estimated_homography: The estimated homography from the first to the second image in
        the pair.
    :param kp_prediction: (K,2) transformed (matched) keypoints from the first image using the
        estimated homography.
    :param kp_ground_truth: (K, 2) transformed (matched) keypoints from the first image using
        the true homography.
    :param corner_prediction: (4, 2) transformed image corners using predicted homography
    :param gorner_ground_truth: (4, 2) transformed image corners using the gt homography
    """
    matches: Matches
    estimated_homography: np.ndarray
    kp_prediction: np.ndarray
    kp_ground_truth: np.ndarray
    corner_prediction: np.ndarray
    corner_ground_truth: np.ndarray

    def __post_init__(self) -> None:
        sc = ShapeCheck()
        assert sc('3, 3').check(self.estimated_homography), sc.why
        assert sc('N,2').check(self.matches.indices), sc.why
        assert sc('N,2').check(self.kp_prediction), sc.why
        assert sc('N,2').check(self.kp_ground_truth), sc.why
        assert sc('4, 2').check(self.corner_prediction), sc.why
        assert sc('4, 2').check(self.corner_ground_truth), sc.why

    @property
    def is_valid(self) -> bool:
        return not bool(np.any(np.isnan(self.estimated_homography)))

    @staticmethod
    def construct_empty(matches: Matches) -> HomographyEstimate:
        return HomographyEstimate(
            matches,
            np.full((3, 3), np.nan),
            np.full((len(matches.indices), 2), np.nan),
            np.full((len(matches.indices), 2), np.nan),
            np.full((4, 2), np.nan),
            np.full((4, 2), np.nan)
        )
