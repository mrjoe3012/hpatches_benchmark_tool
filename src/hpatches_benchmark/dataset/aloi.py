from __future__ import annotations
from dataclasses import dataclass
from hpatches_benchmark.dataset.image_set import ImageSet
from os import path

@dataclass
class ALOI:
    image_sets: list[ImageSet]

    @staticmethod
    def load_aloi(aloi_path: str) -> ALOI:
        aloi_path = path.normpath(aloi_path)
        if not path.isdir(aloi_path):
            raise RuntimeError(f"'{aloi_path}' is not a valid directory.") 
