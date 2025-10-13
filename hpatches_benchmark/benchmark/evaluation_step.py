from dataclasses import dataclass
from typing import Optional

import numpy as np

from hpatches_benchmark.benchmark.homography_metrics import HomographyMetrics
from hpatches_benchmark.dataset.image_with_homography import ImageWithHomography

@dataclass
class EvaluationStep:
    img: ImageWithHomography
    kp1: np.ndarray
    kp2: np.ndarray
    all_match_indices: np.ndarray
    homography: Optional[np.ndarray] = None
    homography_metrics: Optional[HomographyMetrics] = None
