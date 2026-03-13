from __future__ import annotations
from dataclasses import dataclass
from hpatches_benchmark.dataset.image_with_homography import ImageWithHomography
import numpy as np
import cv2

__all__ = ['ImageSet']

@dataclass
class ImageSet:
    name: str
    images: list[ImageWithHomography]

    def resize(self, new_width: int, new_height: int) -> ImageSet:
        result = ImageSet(name=self.name, images=[])
        for img in self.images:
            og_img_h, og_img_w = img.original_img_bgr.shape[:2]
            t_img_h, t_img_w = img.transformed_img_bgr.shape[:2]
            scale1 = [new_width / og_img_w, new_height / og_img_h]
            scale2 = [new_width / t_img_w, new_height / t_img_h]
            scale1_inv_M = np.eye(3, dtype=np.float64)
            scale2_M = scale1_inv_M.copy()
            scale1_inv_M[np.diag_indices(2)] = 1 / np.array(scale1)
            scale2_M[np.diag_indices(2)] = scale2
            scaled_homography = scale2_M @ img.homography @ scale1_inv_M
            new_img = ImageWithHomography(
                filepath=img.filepath,
                original_img_bgr=cv2.resize(img.original_img_bgr, (new_width, new_height)),
                transformed_img_bgr=cv2.resize(img.transformed_img_bgr, (new_width, new_height)),
                homography=scaled_homography
            )
            result.images.append(new_img)
        return result
