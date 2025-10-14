import numpy as np
import cv2

from hpatches_benchmark.benchmark.features import Features

__all__ = ['sift_detector']

def sift_detector(image: np.ndarray):
    sift = cv2.SIFT_create()
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, desc = sift.detectAndCompute(image, None)
    indices = sorted(list(range(len(kp))), key = lambda x: kp[x].response, reverse=True)
    kp = np.array([kp[x].pt for x in indices])
    desc = desc[indices]
    return kp, desc
    