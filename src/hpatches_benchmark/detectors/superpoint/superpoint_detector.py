from importlib import resources
from hpatches_benchmark.detectors.superpoint.superpoint import SuperPoint
import numpy as np
import cv2
import torch

__all__ = ['superpoint_detector']

_model = SuperPoint()

with resources.files("hpatches_benchmark.detectors.superpoint") \
        .joinpath("superpoint_v6_from_tf.pth").open("rb") as f:
    _model.load_state_dict(torch.load(f))
    _model.eval()
    if torch.cuda.is_available():
        _model = _model.to('cuda')

def superpoint_detector(img: np.ndarray):
    if len(img.shape) == 3:
        img = cv2.cvtColor(
            img, cv2.COLOR_BGR2RGB
        )
    elif len(img.shape) != 3:
        img = img[..., None]
    img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
    with torch.no_grad():
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0)
        if torch.cuda.is_available():
            img_tensor = img_tensor.to('cuda')
        kp, des, scores = _model(img_tensor)
        kp_np = kp[0].cpu().detach().numpy()
        des_np = des[0].cpu().detach().numpy()
        scores_np = scores[0].cpu().detach().numpy()
    indices = np.argsort(scores_np)[::-1]
    kp_np = kp_np[indices]
    des_np = des_np[indices]
    return kp_np, des_np
    