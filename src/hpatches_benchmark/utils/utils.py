import numpy as np

__all__ = ['apply_homography']

def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    p_homo = np.concatenate([
        points, np.ones((len(points),)).reshape(-1, 1)
    ], axis=-1)
    p_transformed = p_homo @ H.T
    p_transformed = p_transformed[:, :2] / p_transformed[:, [2]]
    return p_transformed
