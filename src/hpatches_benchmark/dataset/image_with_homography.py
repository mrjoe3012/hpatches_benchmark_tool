from dataclasses import dataclass
from typing import Literal
import numpy as np
from os import path

__all__ = ['ImageWithHomography']

@dataclass
class ImageWithHomography:
    transformed_img_bgr: np.ndarray
    original_img_bgr: np.ndarray
    homography: np.ndarray
    filepath: str

    @property
    def task(self) -> Literal['viewpoint', 'intensity']:
        name = path.basename(path.dirname(self.filepath))
        if name.startswith('v_'): return 'viewpoint'
        elif name.startswith('i_'): return 'intensity'
        else:
            raise RuntimeError(f"Unrecognised task! path: '{self.filepath}'")
