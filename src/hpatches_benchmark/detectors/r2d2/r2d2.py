from r2d2.extract import load_model, get_checkpoint_path, process_img
import torch

__all__ = ['r2d2_detector']

_checkpoint_path = get_checkpoint_path('r2d2_WASF_N16')
_checkpoint = torch.load(_checkpoint_path)
_model = load_model(_checkpoint, verbose=False)
if torch.cuda.is_available():
    _model = _model.cuda()

def r2d2_detector(img):
    xys, desc = process_img(
        img, _model, is_bgr=True
    )
    return xys[:, :2], desc
