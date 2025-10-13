from dataclasses import dataclass
import numpy as np

__all__ = ['ImageWithHomography']

@dataclass
class ImageWithHomography:
    transformed_img_bgr: np.ndarray
    original_img_bgr: np.ndarray
    homography: np.ndarray
    filepath: str

    @property
    def task(self) -> str:
        if self.filepath.startswith('v_'): return 'viewpoint'
        elif self.filepath.startswith('i_'): return 'intensity'
        else:
            raise RuntimeError(f"Unrecognised task! path: '{self.filepath}'")
