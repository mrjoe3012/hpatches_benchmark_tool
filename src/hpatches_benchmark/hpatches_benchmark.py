#!/bin/python3
import logging
from hpatches_benchmark.benchmark.benchmark import run_benchmark
from datetime import datetime
from os import path
import pandas as pd
import numpy as np

from hpatches_benchmark.dataset.hpatches import HPatches
from hpatches_benchmark.detectors.detector import Detector
from hpatches_benchmark.detectors.sift import sift_detector
from hpatches_benchmark.detectors.orb import orb_detector
import cv2
import os
import argparse

__all__ = ['evaluate_detectors', 'main']

def evaluate_detectors(detectors: list[Detector], norms: list[int], output_root: str,
                       hpatches_directory: str) -> None:
    """
    Evaluates the provided detector functions, each taking a BGR image and returning
        keypoints (N,2) and descriptors (N, D).
    :param detectors: Detector functions, detector(img) -> tuple[keypoints, descriptors]
    :param norms: cv2.NORM__L2 or cv2.NORM_HAMMING
    :parma output_root: Where outputs are saved.
    :param hpatches_directory: Path to the extract HPatches sequences dataset.
    """
    timestamp = datetime.now().strftime('%d_%m_%y__%H_%M_%S')
    names = [fn.__name__ for fn in detectors]
    output_root = f'{output_root}/{timestamp}'
    hpatches = HPatches.load_hpatches(hpatches_directory)

    combined_vp_summary = []
    combined_intensity_summary = []
    combined_summary = []

    for detector_fn, norm, name in zip(detectors, norms, names, strict=True):
        output_dir = path.join(output_root, name)
        results = run_benchmark(
            hpatches=hpatches,
            n_kpts=1000,
            detector=detector_fn,
            norm=norm,
            output_dir=output_dir,
            experiment_name=name,
            # N=3
        )
        intensity, viewpoint = results.split_by_task()
        intensity_full = intensity.dataframe
        intensity_summary = intensity.summary_dataframe
        viewpoint_full = viewpoint.dataframe
        viewpoint_summary = viewpoint.summary_dataframe
        summary_both = results.summary_dataframe

        with open(path.join(output_dir, 'summary_viewpoint.txt'), 'w') as f:
            print(viewpoint_summary, file=f)

        with open(path.join(output_dir, 'summary_intensity.txt'), 'w') as f:
            print(intensity_summary, file=f)

        with open(path.join(output_dir, 'summary_both.txt'), 'w') as f:
            print(summary_both, file=f)

        viewpoint_full.to_csv(path.join(output_dir, 'viewpoint_full.csv'))
        intensity_full.to_csv(path.join(output_dir, 'intensity_full.csv'))

        combined_vp_summary.append(viewpoint_summary)
        combined_intensity_summary.append(intensity_summary)
        combined_summary.append(summary_both)

    if len(combined_intensity_summary) > 1:
        output_dir = path.join(output_root, 'all')
        os.makedirs(output_dir)
        viewpoint_summary = pd.concat(combined_vp_summary, axis=0)
        viewpoint_summary.index = names
        intensity_summary = pd.concat(combined_intensity_summary, axis=0)
        intensity_summary.index = names
        summary_both = pd.concat(combined_summary, axis=0)
        summary_both.index = names
        
        with open(path.join(output_dir, 'summary_viewpoint.txt'), 'w') as f:
            print(viewpoint_summary, file=f)

        with open(path.join(output_dir, 'summary_intensity.txt'), 'w') as f:
            print(intensity_summary, file=f)

        with open(path.join(output_dir, 'summary_both.txt'), 'w') as f:
            print(summary_both, file=f)

        viewpoint_summary.to_csv(path.join(output_dir, 'viewpoint_summary.csv'))
        intensity_summary.to_csv(path.join(output_dir, 'intensity_summary.csv'))
        summary_both.to_csv(path.join(output_dir, 'summary_both.csv'))

def main() -> None:
    logging.basicConfig(level='INFO')
    parser = argparse.ArgumentParser()
    parser.add_argument('--hpatches', type=str, required=True, help='Path to HPatches' \
        ' dataset directory.')
    args = parser.parse_args()
    hpatches_dir = args.hpatches
    evaluate_detectors(
        [sift_detector, orb_detector], [cv2.NORM_L2, cv2.NORM_HAMMING],
        output_root='.', hpatches_directory=hpatches_dir
    )

if __name__ == '__main__': main()
