import numpy as np
from typing import Callable

from tabulate import tabulate
from hpatches_benchmark.benchmark.benchmark_result import BenchmarkResult
from hpatches_benchmark.benchmark.evaluation_step import EvaluationStep
from hpatches_benchmark.benchmark.homography_metrics import HomographyMetrics
from hpatches_benchmark.dataset.hpatches import HPatches
import cv2
from tqdm import tqdm
from hpatches_benchmark.dataset.image_set import ImageSet
from hpatches_benchmark.dataset.image_with_homography import ImageWithHomography
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from typing import Optional
from os import path
from hpatches_benchmark.utils.logger import logger
import os
import pickle

__all__ = ['evaluate']

kp_marker_size = 45

def sift_detector(image: np.ndarray, max_keypoints: int) -> tuple[np.ndarray, np.ndarray]:
    sift = cv2.SIFT_create()
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(image, None)
    indices = sorted(list(range(len(kp))), key = lambda x: kp[x].response, reverse=True)[:max_keypoints]
    kp = np.array([kp[x].pt for x in indices])
    desc = desc[indices]
    return kp, desc

def apply_homography(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    p_homo = np.concatenate([
        points, np.ones((len(points),)).reshape(-1, 1)
    ], axis=-1)
    p_transformed = p_homo @ H.T
    p_transformed = p_transformed[:, :2] / p_transformed[:, [2]]
    return p_transformed

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
                homography_gt: np.ndarray) -> None:
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
                 matches: np.ndarray) -> None:
    height, width = plot_imgs_side_by_side(ax, img1, img2)
    correspondences = LineCollection(
        np.stack([kp1[matches[:, 0]], kp2[matches[:, 1]] + [img1.shape[1], 0]], axis=-2),
        color='green'
    )
    ax.add_collection(correspondences)
    ax.scatter(
        kp1[:, 0], kp1[:, 1],
        facecolor='none', color='blue', s=kp_marker_size
    )
    ax.scatter(
        kp2[:, 0] + img1.shape[1], kp2[:, 1],
        facecolor='none', color='blue', s=kp_marker_size
    )

def detect_and_match(img: ImageWithHomography, detector, norm, n_kpts: int,
                     img_size: Optional[tuple[int, int]]) -> EvaluationStep:
    # resize imgs if necessary
    if img_size is None:
        og_img = img.original_img_bgr
        transformed_img = img.transformed_img_bgr
        scaling1 = [1, 1]
        scaling2 = [1, 1]
    else:
        og_img = cv2.resize(img.original_img_bgr, img_size)
        transformed_img = cv2.resize(img.transformed_img_bgr, img_size)
        scaling1 = [img.original_img_bgr.shape[1] / img_size[0], img.original_img_bgr.shape[0] / img_size[1]]
        scaling2 = [img.transformed_img_bgr.shape[1] / img_size[0], img.transformed_img_bgr.shape[0] / img_size[1]]
    matcher = cv2.BFMatcher_create(norm)
    # detect features
    kp1, des1 = detector(og_img, n_kpts)
    kp2, des2 = detector(transformed_img, n_kpts)
    # match features
    matches = matcher.knnMatch(des1, des2, k=2)
    # use lowe's ratio test
    matches = [m1 for m1, m2 in matches if m1.distance < 0.75 * m2.distance]
    # sort matches by distance
    matches = sorted(matches, key=lambda m: m.distance)
    # form 2d array of indices
    match_indices = np.array([[x.queryIdx, x.trainIdx] for x in matches])
    return EvaluationStep(
        img, kp1 * scaling1, kp2 * scaling2, match_indices
    )

def compute_homography(eval_step: EvaluationStep, n_kpts: int) -> np.ndarray:
    # compute homography
    pred_homography = None
    try:
        pred_homography, mask = cv2.findHomography(
            eval_step.kp1[eval_step.all_match_indices[:n_kpts, 0]].astype(np.float32).reshape(-1, 1, 2),
            eval_step.kp2[eval_step.all_match_indices[:n_kpts, 1]].astype(np.float32).reshape(-1, 1, 2),
            method=cv2.RANSAC
        )
    except cv2.error:
        pass
    if pred_homography is None:
        logger.warning(f"find_homography failed for {eval_step.img.filepath}")
        pred_homography = np.eye(3, dtype=np.float32)
    return pred_homography

def evaluate_homography(eval_step: EvaluationStep, homography: np.ndarray, n_kpts: int,
                        img_size: Optional[tuple[int, int]]) -> HomographyMetrics:
    if img_size:
        width, height = img_size
    else:
        height, width = eval_step.img.original_img_bgr.shape[:2]
    # points representing corners of the image
    corners = np.array([
        0, 0,
        width, 0,
        0, height,
        width, height
    ]).reshape(-1, 2)
    # apply estim homography and gt homography to corners of img
    corners_pred = apply_homography(
        corners, homography
    )
    corners_gt = apply_homography(
        corners, eval_step.img.homography
    )
    corner_distances = np.linalg.norm(corners_gt - corners_pred, axis=-1)
    # and now to keypoints
    matched_indices = eval_step.all_match_indices[:, 0]
    kp2_pred = apply_homography(
        eval_step.kp1, eval_step.homography
    )
    kp2_gt = apply_homography(
        eval_step.kp1, eval_step.img.homography
    )
    kp_distances = np.linalg.norm(kp2_pred[matched_indices] - kp2_gt[matched_indices], axis=-1)
    return HomographyMetrics.compute(kp_distances, corner_distances, n_kpts)

def evaluate(root_path: str, n_kpts: int = 1000,
             detector: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]] = sift_detector,
             norm = cv2.NORM_L2, progress_bar: bool = True,
             output_dir: Optional[str] = '.',
             high_quality_plot: bool = True,
             img_size: Optional[tuple[int, int]] = None) -> BenchmarkResult:
    if img_size is None:
        img_size = (640, 480) # consistent image size is important to ensure benchmark does not bias towards certain images
    pbar = tqdm if progress_bar else lambda x, *kwargs: x
    hpatches = HPatches.load_hpatches(root_path)
    viewpoint_eval_steps: list[EvaluationStep] = []
    intensity_eval_steps: list[EvaluationStep] = []
    for img_set in pbar(hpatches.image_sets[:35]):
        img_set: ImageSet
        intensity_img = img_set.name.startswith('i_')
        viewpoint_img = img_set.name.startswith('v_')
        if intensity_img or viewpoint_img:
            for img_with_homo in img_set.images:
                # run detector
                eval_step = detect_and_match(
                    img_with_homo,
                    detector,
                    norm,
                    n_kpts,
                    img_size
                )
                # compute repeatability
                # rep_metrics = evaluate_repeatability()
                if viewpoint_img:
                    # find homography
                    homography = compute_homography(eval_step, n_kpts)
                    eval_step.homography = homography
                    # compute metrics
                    homo_metrics = evaluate_homography(eval_step, homography, n_kpts, img_size)
                    eval_step.homography_metrics = homo_metrics
                    viewpoint_eval_steps.append(eval_step)
        else:
            logger.warning(f"HPatches folder name didn't start with v_ or i_! {img_set.name=}")
    # save the three best and worst
    if output_dir is not None:
        output_dir = path.normpath(output_dir)
        if path.exists(output_dir) and not path.isdir(output_dir):
            raise RuntimeError(f"{output_dir} is not a directory!")
        os.makedirs(output_dir, exist_ok=True)
        loc_errs = np.array([np.mean(x.homography_metrics.kp_distances) for x in viewpoint_eval_steps])
        best_indices = np.argsort(loc_errs)[:3]
        worst_indices = np.argsort(loc_errs)[-3:]
        for arr, prefix in zip([best_indices, worst_indices], ['best', 'worst']):
            for idx in arr:
                fig, axes = plt.subplots(2, 1, figsize=(48, 36))
                plot_homography(
                    axes[0],
                    viewpoint_eval_steps[idx].img.original_img_bgr,
                    viewpoint_eval_steps[idx].img.transformed_img_bgr,
                    viewpoint_eval_steps[idx].kp1,
                    viewpoint_eval_steps[idx].homography,
                    viewpoint_eval_steps[idx].img.homography
                )
                plot_matches(
                    axes[1],
                    viewpoint_eval_steps[idx].img.original_img_bgr,
                    viewpoint_eval_steps[idx].img.transformed_img_bgr,
                    viewpoint_eval_steps[idx].kp1,
                    viewpoint_eval_steps[idx].kp2,
                    viewpoint_eval_steps[idx].all_match_indices
                )
                if high_quality_plot:
                    kwargs = {
                        'bbox_inches' : 'tight',
                        'dpi' : 300
                    }
                else:
                    kwargs = {}
                plt.savefig(path.join(output_dir, f'{prefix}_{idx}.png', **kwargs))
                plt.close(fig)
        benchmark_result = BenchmarkResult.compute(
            viewpoint_eval_steps, intensity_eval_steps
        )
        if output_dir:
            with open(path.join(output_dir, 'benchmark_result.p'), 'wb') as f:
                pickle.dump(benchmark_result, f)
            with open(path.join(output_dir, 'summary.txt'), 'w') as f:
                f.write(benchmark_result.table())
        return benchmark_result