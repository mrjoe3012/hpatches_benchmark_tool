from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class HomographyMetrics:
    kp_distances: np.ndarray
    img_corner_distances: np.ndarray
    repeatability_distances: np.ndarray
    n_kpts: int

    @staticmethod
    def compute(kp_dist: np.ndarray, img_corner_dist: np.ndarray, n_kpts: int) -> HomographyMetrics:
        return HomographyMetrics(
            kp_dist, img_corner_dist, n_kpts
        )
