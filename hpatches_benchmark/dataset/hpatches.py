from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from hpatches_benchmark.dataset.image_with_homography import ImageWithHomography
from hpatches_benchmark.dataset.image_set import ImageSet
from hpatches_benchmark.utils.logger import logger
from os import path
import os
import cv2
import numpy as np
from tqdm import tqdm

__all__ = ['HPatches']

@dataclass
class HPatches:
    image_sets: list[ImageSet]

    @staticmethod
    def load_hpatches(root_directory: str, progress_bar: bool = True) -> HPatches:
        pbar = tqdm if progress_bar else lambda x, **kwargs: x
        subdirectories = [
            path.join(root_directory, x) for x in os.listdir(root_directory)
                if path.isdir(path.join(root_directory, x))
        ]
        image_sets = []
        for subdirectory in pbar(subdirectories):
            name = path.basename(subdirectory)
            og_img_path = path.join(subdirectory, '1.ppm') 
            og_img = cv2.imread(og_img_path)
            if og_img is None:
                logger.warning(f"{subdirectory} did not contain {og_img}")
                continue
            i = 2
            images = []
            homographies = []
            filepaths = []
            while True:
                img_path = path.join(subdirectory, f'{i}.ppm')
                if not path.isfile(img_path):
                    break
                img = cv2.imread(img_path)
                if img is None:
                    break
                homo_path = path.join(subdirectory, f'H_1_{i}')
                try:
                    homo = np.loadtxt(homo_path)
                except:
                    logger.warning(f"Failed to read homography {homo_path}")
                    continue
                finally:
                    i += 1
                images.append(img)
                homographies.append(homo)
                filepaths.append(img_path)
            image_sets.append(ImageSet(name, [ImageWithHomography(img, og_img, homo, fp)
                                             for img, homo, fp in zip(images, homographies, filepaths)]))
        return HPatches(image_sets=image_sets) 
