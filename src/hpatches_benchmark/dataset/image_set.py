from __future__ import annotations
from dataclasses import dataclass
from hpatches_benchmark.dataset.image_with_homography import ImageWithHomography
import numpy as np
import cv2
from numpy.random import RandomState

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

    def add_noise(self, noise_sigma: float, blur_sigma: float, rng: np.random.RandomState) -> ImageSet:
        """
        Additive Gaussian noise + Gaussian blur to all images.

        noise_sigma: standard deviation of additive Gaussian noise (0–255 scale).
        blur_sigma : standard deviation of Gaussian blur kernel (pixels).
        rng        : numpy RandomState used for reproducible noise sampling.
        """

        corrupted_imgs = []

        for img_w_homo in self.images:
            uncorrupted_img = img_w_homo.original_img_bgr
            uncorrupted_img_transformed = img_w_homo.transformed_img_bgr

            # Ensure we work in float32 for arithmetic
            img = uncorrupted_img.astype(np.float32)
            img_t = uncorrupted_img_transformed.astype(np.float32)

            # --- Additive Gaussian noise (sensor / readout noise proxy) ---
            if noise_sigma > 0:
                noise = rng.normal(0.0, noise_sigma, img.shape).astype(np.float32)
                noise_t = rng.normal(0.0, noise_sigma, img_t.shape).astype(np.float32)
                img = img + noise
                img_t = img_t + noise_t

            # Clip to valid range before blur
            img = np.clip(img, 0, 255)
            img_t = np.clip(img_t, 0, 255)

            # --- Gaussian blur (defocus / motion blur proxy) ---
            if blur_sigma > 0:
                # Choose kernel size from sigma (odd, at least 3)
                ksize = int(blur_sigma * 6 + 1)
                if ksize % 2 == 0:
                    ksize += 1
                ksize = max(3, ksize)

                img = cv2.GaussianBlur(img, (ksize, ksize), blur_sigma)
                img_t = cv2.GaussianBlur(img_t, (ksize, ksize), blur_sigma)

            # Final clip and convert back to uint8
            corrupted_img = np.clip(img, 0, 255).astype(np.uint8)
            corrupted_img_transformed = np.clip(img_t, 0, 255).astype(np.uint8)

            corrupted_imgs.append(ImageWithHomography(
                corrupted_img_transformed,
                corrupted_img,
                img_w_homo.homography,
                img_w_homo.filepath
            ))

        return ImageSet(self.name, corrupted_imgs)