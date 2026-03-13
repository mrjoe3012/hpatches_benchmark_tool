
from hpatches_benchmark.benchmark.benchmark_result import BenchmarkResult
from hpatches_benchmark.benchmark.features import Features
from hpatches_benchmark.benchmark.homography_estimate import HomographyEstimate
from hpatches_benchmark.benchmark.homography_evaluation import HomographyEvaluation
from hpatches_benchmark.benchmark.homography_evaluation_iou import HomographyEvaluationIOU
from hpatches_benchmark.benchmark.matches import Matches
from hpatches_benchmark.benchmark.matching_score_evaluation import MatchingScoreEvaluation
from hpatches_benchmark.benchmark.mma_evaluation import MMAEvaluation
from hpatches_benchmark.benchmark.repeatability_evaluation import RepeatabilityEvaluation
from hpatches_benchmark.dataset.hpatches import HPatches
from tqdm import tqdm
from hpatches_benchmark.dataset.image_set import ImageSet
from hpatches_benchmark.dataset.image_with_homography import ImageWithHomography
from typing import Optional
from hpatches_benchmark.detectors.detector import Detector
from hpatches_benchmark.utils.logger import logger
from hpatches_benchmark.utils.utils import apply_homography
from os import path
from hpatches_benchmark.benchmark.visualisation import *
from shapely.geometry import Polygon
from shapely.errors import GEOSException
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

__all__ = ['run_benchmark']

_DEFAULT_IOU_EPSILON = np.linspace(0, 1, 1_000)

def get_features(img: ImageWithHomography, detector: Detector) -> Features:
    og_img = img.original_img_bgr
    transformed_img = img.transformed_img_bgr
    # detect features
    kp1, des1 = detector(og_img)
    kp2, des2 = detector(transformed_img)
    return Features(
        img,
        kp1, des1,
        kp2, des2
    )

def get_matches(features: Features, norm: int, n_kpts: int) -> Matches:
    matcher = cv2.BFMatcher_create(norm, crossCheck=True)
    # match features
    matches = matcher.match(
        features.descriptors_1[:n_kpts],
        features.descriptors_2[:n_kpts],
    )
    # sort matches by distance
    matches = sorted(matches, key=lambda m: m.distance)
    # form 2d array of indices
    match_indices = np.array([[x.queryIdx, x.trainIdx] for x in matches])
    return Matches(
        features,
        match_indices.reshape(-1, 2)  # handle case for no matches -> (0, 2)
    )

def get_homography(matches: Matches) -> HomographyEstimate:
    img1_shape = matches.features.img.original_img_bgr.shape
    img1_height, img1_width = img1_shape[:2]
    # compute homography
    pred_homography = None
    match_indices = matches.indices
    kp1, kp2 = matches.features.keypoints_1, matches.features.keypoints_2
    kp1 = kp1[match_indices[:, 0]]
    kp2 = kp2[match_indices[:, 1]]
    try:
        pred_homography, mask = cv2.findHomography(
            kp1.astype(np.float32).reshape(-1, 1, 2),
            kp2.astype(np.float32).reshape(-1, 1, 2),
            method=cv2.RANSAC,
        )
    except cv2.error:
        pass
    if pred_homography is None:
        logger.warning(f"find_homography failed for {matches.features.img.filepath}")
        pred_homography = np.full((3, 3), np.nan)
    H_pred = pred_homography
    H_true = matches.features.img.homography
    img_corners = np.array([
        1, 1,
        img1_width, 1,
        1, img1_height,
        img1_width, img1_height,
    ]).reshape(4, 2) - 1
    img_corners_pred = apply_homography(
        img_corners, H_pred
    )
    img_corners_true = apply_homography(
        img_corners, H_true
    )
    kp1_t_pred = apply_homography(
        kp1, H_pred
    )
    kp1_t_true = apply_homography(
        kp1,
        H_true
    )
    return HomographyEstimate(
        matches,
        pred_homography,
        kp1_t_pred,
        kp1_t_true,
        img_corners_pred,
        img_corners_true
    )

def evaluate_homography(homography_estimate: HomographyEstimate, epsilon: np.ndarray) -> HomographyEvaluation:
    if np.any(np.isnan(homography_estimate.estimated_homography)):
        return HomographyEvaluation.construct_empty(
            homography_estimate.matches, epsilon
        )
    mean_corner_error = np.mean(
        np.linalg.norm(
            homography_estimate.corner_ground_truth - homography_estimate.corner_prediction,
            axis=-1
        ),
        keepdims=True
    ) 
    correct = (mean_corner_error[None, :] <= epsilon[:, None]).astype(np.float64).squeeze(-1)
    localisation_errors = np.linalg.norm(
        homography_estimate.kp_ground_truth - homography_estimate.kp_prediction,
        axis=-1,
    )
    with np.errstate(divide='ignore', invalid='ignore'):
        included_in_mean = localisation_errors[None, :] <= epsilon[:, None]
        sum_localisation_err = included_in_mean @ localisation_errors
        counts = np.count_nonzero(included_in_mean, axis=-1)
        mean_localisation_err = sum_localisation_err / counts
        mean_localisation_err[counts == 0] = np.nan
    return HomographyEvaluation(
        homography_estimate,
        epsilon,
        mean_localisation_err,
        correct,
        float(mean_corner_error)
    )

def evaluate_homography_iou(homography_estimate: HomographyEstimate,
                            epsilon_iou: Optional[np.ndarray] = None) -> HomographyEvaluationIOU:
    if np.any(np.isnan(homography_estimate.estimated_homography)):
        return HomographyEvaluationIOU.construct_empty(
            homography_estimate.matches, epsilon_iou
        )
    if epsilon_iou is None:
        epsilon_iou = _DEFAULT_IOU_EPSILON.copy()
    # permute corners so that they go CW around the shell
    permutation = [0, 2, 3, 1]
    try:
        ground_truth_box = Polygon(homography_estimate.corner_ground_truth[permutation]) 
        predicted_box = Polygon(homography_estimate.corner_prediction[permutation])
        if predicted_box.is_valid:
            intersection = ground_truth_box.intersection(predicted_box).area
            union = ground_truth_box.union(predicted_box).area
            iou = intersection / union
        else:
            iou = 0.0
        correct_homo = iou >= epsilon_iou
    except GEOSException as e:
        # topological error means that the homography was invalid
        # set correctness to 0 and IOU to 0
        correct_homo = np.full(epsilon_iou.shape, False)
        iou = 0.0
    return HomographyEvaluationIOU(
        homography_estimate,
        epsilon_iou,
        correct_homo,
        iou
    )

def evaluate_repeatability(features: Features, epsilon: np.ndarray,
                           n_kpts: int, img_size_wh: tuple[int, int]) -> RepeatabilityEvaluation:
    width, height = img_size_wh
    kp1, kp2 = features.keypoints_1[:n_kpts], features.keypoints_2[:n_kpts]
    if len(kp1) == 0 or len(kp2) == 0:
        return RepeatabilityEvaluation.construct_empty(features, epsilon, n_kpts)
    H = features.img.homography
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        logger.warning(f"Homography from {features.img.filepath} was not invertable.")
        return RepeatabilityEvaluation.construct_empty(features, epsilon, n_kpts)
    # transform both keypoints by ground truth
    kp1_t = apply_homography(kp1, H) 
    kp2_t = apply_homography(kp2, H_inv)
    kp1 = kp1
    kp2 = kp2
    # keep only points in mutually shared image region
    kp1_mask = \
        (np.all(kp1_t >= [0, 0], axis=-1)) & (np.all(kp1_t < [width, height], axis=-1))
    kp2_mask = \
        (np.all(kp2_t >= [0, 0], axis=-1)) & (np.all(kp2_t < [width, height], axis=-1))
    kp1, kp1_t = kp1[kp1_mask], kp1_t[kp1_mask]
    kp2, kp2_t = kp2[kp2_mask], kp2_t[kp2_mask]
    if len(kp1) == 0 or len(kp2) == 0:
        return RepeatabilityEvaluation.construct_empty(features, epsilon, n_kpts)
    # symmetrical nearest neighbour matching
    dist1 = np.linalg.norm(
        kp2[None] - kp1_t[:, None],
        axis=-1
    )
    dist2 = np.linalg.norm(
        kp1[None] - kp2_t[:, None],
        axis=-1
    )
    symmetrically_matched = \
        np.argmin(dist1, axis=-1) == np.argmin(dist2, axis=0)
    # check against correctness thresholds
    within_threshold = np.min(dist1, axis=-1)[None] <= epsilon[:, None]
    repeatable = within_threshold & symmetrically_matched[None]
    repeatability = np.count_nonzero(repeatable, axis=-1) / min(len(kp1), len(kp2))
    return RepeatabilityEvaluation(
        features, epsilon, repeatability, n_kpts
    )

def evaluate_mma(matches: Matches, epsilon: np.ndarray) -> MMAEvaluation:
    kp1, kp2 = matches.features.keypoints_1, matches.features.keypoints_2
    H = matches.features.img.homography
    if len(matches.indices) == 0:
        return MMAEvaluation.construct_empty(matches, epsilon)
    kp1_matches = kp1[matches.indices[:, 0]]
    kp2_matches = kp2[matches.indices[:, 1]]
    kp1_w = apply_homography(kp1_matches, H)
    distances = np.linalg.norm(
        kp2_matches - kp1_w,
        axis=-1
    )
    correct = distances <= np.expand_dims(epsilon, 1)
    mma = np.count_nonzero(correct, axis=-1) / len(distances)
    return MMAEvaluation(
        matches,
        epsilon,
        mma
    )

def evaluate_matching_score(matches: Matches, epsilon: np.ndarray, n_kpts: int,
                            img_size: tuple[int, int]) -> MatchingScoreEvaluation:
    width, height = img_size
    def comp_m_score(kp1, kp2, H, match_indices):
        kp1_w = apply_homography(
            kp1, H
        )
        in_view = np.all(kp1_w >= [0, 0], axis=-1) & np.all(kp1_w < [width, height], axis=-1)
        if np.count_nonzero(in_view) == 0:
            return np.full(epsilon.shape, 0)
        matches_to_use = in_view[match_indices[:, 0]]
        kp1_m = kp1_w[match_indices[matches_to_use, 0]]
        kp2_m = kp2[match_indices[matches_to_use, 1]]
        dist = np.linalg.norm(kp1_m - kp2_m, axis=-1)
        correct = dist <= np.expand_dims(epsilon, 1)
        return np.count_nonzero(correct, axis=-1) / np.count_nonzero(in_view)

    kp1, kp2 = matches.features.keypoints_1, matches.features.keypoints_2
    kp1 = kp1[:n_kpts]
    kp2 = kp2[:n_kpts]
    H = matches.features.img.homography
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        logger.error(f"Unable to invert homography from {matches.features.img.filepath}")
        return MatchingScoreEvaluation.construct_empty(matches, epsilon)
    scores1 = comp_m_score(
        kp1, kp2, H, matches.indices
    )
    scores2 = comp_m_score(
        kp2, kp1, H_inv, matches.indices[:, ::-1]
    )
    return MatchingScoreEvaluation(
        matches, epsilon, (scores1 + scores2) / 2
    )

def make_plots(output_dir: str, homos: list[HomographyEvaluation],
               n_kpts: int, N: int  = 3) -> None:
    """
    Plot the N worst and N best from homography estimation.
    """
    sorted_indices = sorted(
        list(range(len(homos))),
        key = lambda x: homos[x].mean_corner_error
    )
    best_indices, worst_indices = sorted_indices[:N], sorted_indices[-N:]
    for arr, prefix in zip([best_indices, worst_indices], ['best', 'worst']):
        for idx in arr:
            img_path = path.join(output_dir, f'homo_{prefix}_{idx}.png')
            homo = homos[idx].homography_estimate
            matches = homo.matches
            features = homo.matches.features
            img = homo.matches.features.img
            fig, axes = plt.subplots(2, 1, figsize=(12, 9))
            plot_homography(
                axes[0], img.original_img_bgr, img.transformed_img_bgr,
                features.keypoints_1[:n_kpts], homo.estimated_homography,
                img.homography
            )
            plot_matches(
                axes[1], img.original_img_bgr, img.transformed_img_bgr,
                features.keypoints_1[:n_kpts], features.keypoints_2[:n_kpts], matches.indices
            )
            fig.savefig(img_path)
            plt.close(fig)

def run_benchmark(hpatches: HPatches, n_kpts: int,
             detector: Detector, norm = cv2.NORM_L2,
             output_dir: Optional[str] = '.',
             img_size_wh: tuple[int, int] = (640, 480),
             progress_bar: bool = True,
             N: Optional[int] = None,
             epsilon: Optional[list[float]] = None,
             experiment_name: str = '') -> BenchmarkResult:
    """
    Run the benchmark on the hpatches dataset, keeping a maximum of n_kpts from the detector.
    :param hpatches: The dataset instance.
    :param n_kpts: This number of the best keypoints are kept for the benchmark.
    :param detector: A function taking a BGR image and returning the keypoints (N,2), and
        descriptors (N, D) where the arrays are parallel and sorted by best keypoints first.
    :param norm: OpenCV norm, such as cv2.NORM_L2, which defines how descriptors should be
        compared.
    :param output_dir: Where the plots / tables will be saved. Leave as None to save nothing
        to disk.
    :param img_size_wh: Image size (width and height) in pixels that each image in the dataset
        is resized to before running the detector.
    :param progress_bar: True to show a progress bar while running the benchmark.
    :param N: The number of image sets to process. Leave as None to use them all.
    :param epsilon: The thresholds to use for computing metrics. Leave as None for default
        of 1, 3, 5
    """
    if epsilon is None:
        epsilon = [1, 3, 5]
    epsilon = np.array(epsilon)
    pbar = tqdm if progress_bar else lambda x, *kwargs: x
    if N is None:
        N = len(hpatches.image_sets)
    homo = []
    homo_iou = []
    rep = []
    mma = []
    m_score = []
    for img_set in pbar(hpatches.image_sets[:N], desc=f'Benchmarking {experiment_name}'):
        img_set: ImageSet
        img_set = img_set.resize(img_size_wh[0], img_size_wh[1])
        for img_with_homo in img_set.images:
            features = get_features(
                img_with_homo,
                detector,
            )
            matches = get_matches(
                features,
                norm,
                n_kpts,
            )
            mma_evaluation = evaluate_mma(
                matches, epsilon
            )
            m_score_evalaution = evaluate_matching_score(
                matches, epsilon, n_kpts, img_size_wh
            )
            if len(matches.indices) >= 4:
                homography_estimate = get_homography(
                    matches
                )
                if homography_estimate.is_valid:
                    homography_evaluation = evaluate_homography(
                        homography_estimate, epsilon
                    )
                    homography_evaluation_iou = evaluate_homography_iou(
                        homography_estimate  # leave epsilon None, different epsilon is used for IOU
                    )
                else:
                    homography_evaluation = HomographyEvaluation.construct_empty(
                        matches, epsilon
                    )
                    homography_evaluation_iou = HomographyEvaluationIOU.construct_empty(
                        matches, _DEFAULT_IOU_EPSILON
                    )
            else:
                logger.warning(f"Less than four matches for '{img_with_homo.filepath}'!")
                homography_evaluation = HomographyEvaluation.construct_empty(
                    matches, epsilon
                )
            repeatability_evaluation = evaluate_repeatability(
                features, epsilon, n_kpts, img_size_wh
            )
            homo.append(homography_evaluation)
            homo_iou.append(homography_evaluation_iou)
            rep.append(repeatability_evaluation)
            mma.append(mma_evaluation)
            m_score.append(m_score_evalaution)
    if output_dir is not None:
        if path.exists(output_dir) and not path.isdir(output_dir):
            logger.warning(f"'{output_dir}' exists and is not a directory.")
        else:
            os.makedirs(output_dir, exist_ok=True)
            homo_viewpoint_only = [
                x for x in homo
                    if x.homography_estimate.matches.features.img.task == 'viewpoint'
            ]
            make_plots(output_dir, homo_viewpoint_only, n_kpts, N=30)
    return BenchmarkResult(hpatches, homo, homo_iou, rep, mma, m_score)
