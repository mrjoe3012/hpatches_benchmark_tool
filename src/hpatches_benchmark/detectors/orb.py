import cv2
import numpy as np

__all__ = ['orb_detector']

def orb_detector(image: np.ndarray):
    orb = cv2.ORB_create()
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp = orb.detect(image, None)
    kp, desc = orb.compute(image, kp)
    indices = sorted(list(range(len(kp))), key = lambda x: kp[x].response, reverse=True)
    kp = np.array([kp[x].pt for x in indices])
    desc = desc[indices]
    return kp, desc
