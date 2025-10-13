from dataclasses import dataclass
from hpatches_benchmark.dataset.image_with_homography import ImageWithHomography

__all__ = ['ImageSet']

@dataclass
class ImageSet:
    name: str
    images: list[ImageWithHomography]
