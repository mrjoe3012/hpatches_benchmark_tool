from dataclasses import dataclass

import numpy as np

from hpatches_benchmark.benchmark.features import Features

__all__ = ['Matches']

@dataclass
class Matches:
    """
    Result from matching features between image pair.
    :param features: Features from an image pair.
    :param matches: (N,2) array of matches, sorted by best match first. matches[i] = [j, k]
        where feature j from features_a is matches to feature k from features_b and match[l] is
        better or as good as any match[m] for m > l. N <= the number of keypoints in the first
        image of the pair.
    """
    features: Features
    indices: np.ndarray[np.int64, 2]

    @property
    def num_points_used(self) -> int:
        """
        Returns the number of keypoints used in matching. <= total number of keypoints.
        """
        return self.indices.shape[0]

    def __post_init__(self) -> None:
        assert self.indices.shape[0] <= len(self.features.keypoints_1), "There cannot be more " \
            "matches than features."
        assert self.indices.shape[1] == 2, "There must be two indices per match."
    
    def __len__(self) -> int:
        return self.indices.shape[0]
