from dataclasses import dataclass
from ndshapecheck import ShapeCheck

import numpy as np

from hpatches_benchmark.dataset.image_with_homography import ImageWithHomography

__all__ = ['Features']

@dataclass
class Features:
    """
    :param img: The image pair that the data comes from. Keypoints are sorted by best keypoint
        first.
    :param keypoints_1: The keypoints detected from the first image (N,2).
    :param keypoints_2: The keypoints detected from the second image (M,2).
    :param descriptors_1: The descriptors, parallel to keypoints_1. (N,D)
    :param descriptors_2: The descriptors, parallel to keypoints_2. (M,D)
    """
    img: ImageWithHomography
    keypoints_1: np.ndarray
    descriptors_1: np.ndarray
    keypoints_2: np.ndarray
    descriptors_2: np.ndarray

    def __post_init__(self) -> None:
        sc = ShapeCheck()
        assert sc('N,2').check(self.keypoints_1), sc.why
        assert sc('M,2').check(self.keypoints_2), sc.why
        assert sc('N,D').check(self.descriptors_1), sc.why
        assert sc('M,D').check(self.descriptors_2), sc.why
