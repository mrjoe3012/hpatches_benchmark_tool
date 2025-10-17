import cv2
import numpy as np
from silk import load_model
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
import torch

__all__ = ['silk_detector']

_model = load_model(nms=0.0)
_model = _model.eval()
if torch.cuda.is_available():
    model = _model.cuda()

def silk_detector(img: np.ndarray):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_tensor = torch.tensor(img / 255.0, dtype=torch.float32)[None, None]
    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
    with torch.no_grad():
        kps, des = model(img_tensor)
        kps = from_feature_coords_to_image_coords(model, kps)
        kps = kps[0].detach().cpu().numpy()
        des = des[0].detach().cpu().numpy()
    kp = kps[:, [1, 0]]
    scores = kps[:, -1]
    sorted_indices = np.argsort(scores)[::-1]
    return kp[sorted_indices], des[sorted_indices]
