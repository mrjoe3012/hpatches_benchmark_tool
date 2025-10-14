from typing import Optional
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
import numpy as np

from hpatches_benchmark.utils.utils import apply_homography

_kp_marker_size = 45

__all__ = ['plot_imgs_side_by_side', 'plot_homography', 'plot_matches']

def plot_imgs_side_by_side(ax: Axes, img1: np.ndarray, img2: np.ndarray) -> tuple[int, int]:
    """
    :returns: height, width of the canvas
    """
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    height = max(img1.shape[0], img2.shape[0])
    width = img1.shape[1] + img2.shape[1]
    final_img = np.zeros(
        (height, width, img1.shape[2]),
        dtype=img1.dtype
    )
    final_img[:img1.shape[0], :img1.shape[1]] = img1
    final_img[:img2.shape[0], img1.shape[1]:] = img2
    ax.imshow(final_img)
    return height, width

def plot_homography(ax: Axes, img1: np.ndarray, img2: np.ndarray,
                kp1: np.ndarray,
                homography_pred: np.ndarray,
                homography_gt: np.ndarray,
                kp_marker_size: Optional[int] = None) -> None:
    if kp_marker_size is None:
        kp_marker_size = _kp_marker_size
    height, width = plot_imgs_side_by_side(ax, img1, img2)
    kp_gt = apply_homography(kp1, homography_gt)
    kp_pred = apply_homography(kp1, homography_pred)
    kp_gt_valid = np.all(
        (kp_gt > [0, 0]) & (kp_gt < [height, width]),
        axis=-1
    )
    kp_pred_valid = np.all(
        (kp_pred > [0, 0]) & (kp_pred < [height, width]),
        axis=-1
    )
    ax.scatter(
        kp1[:, 0], kp1[:, 1], facecolor='none', color='blue', label='Image 1 keypoints', s=kp_marker_size
    )
    correspondences = LineCollection(
        np.stack([kp_pred[kp_pred_valid & kp_gt_valid], kp_gt[kp_pred_valid & kp_gt_valid]], axis=-2) \
            + [img1.shape[1], 0],
        color='black'
    )
    ax.add_collection(correspondences)
    ax.scatter(
        kp_gt[kp_gt_valid, 0] + img1.shape[1], kp_gt[kp_gt_valid, 1], facecolor='none', color='red', label='Image 2 Keypoints (Ground Truth)',
        s=kp_marker_size
    )
    ax.scatter(
        kp_pred[kp_pred_valid, 0] + img1.shape[1], kp_pred[kp_pred_valid, 1], marker='+', facecolor='blue', color='blue', label='Image 2 Keypoints (Predicted)',
        s=kp_marker_size
    )
    ax.legend()

def plot_matches(ax: Axes, img1: np.ndarray, img2: np.ndarray,
                 kp1: np.ndarray, kp2: np.ndarray,
                 matches: np.ndarray,
                 kp_marker_size: Optional[int] = None) -> None:
    if kp_marker_size is None:
        kp_marker_size = _kp_marker_size
    plot_imgs_side_by_side(ax, img1, img2)
    correspondences = LineCollection(
        np.stack([kp1[matches[:, 0]], kp2[matches[:, 1]] + [img1.shape[1], 0]], axis=-2),
        color='green'
    )
    ax.add_collection(correspondences)
    kp1_mask = np.full((kp1.shape[0],), False)
    kp1_mask[matches[:, 0]] = True
    kp2_mask = np.full((kp2.shape[0],), False)
    kp2_mask[matches[:, 1]] = True
    ax.scatter(
        kp1[kp1_mask, 0], kp1[kp1_mask, 1],
        facecolor='none', color='blue', s=kp_marker_size
    )
    ax.scatter(
        kp1[~kp1_mask, 0], kp1[~kp1_mask, 1],
        facecolor='none', color='red', s=kp_marker_size
    )
    ax.scatter(
        kp2[kp2_mask, 0] + img1.shape[1], kp2[kp2_mask, 1],
        facecolor='none', color='blue', s=kp_marker_size
    )
    ax.scatter(
        kp2[~kp2_mask, 0] + img1.shape[1], kp2[~kp2_mask, 1],
        facecolor='none', color='red', s=kp_marker_size
    )
