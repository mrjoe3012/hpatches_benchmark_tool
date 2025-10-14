from typing import Callable
import numpy as np

Detector = Callable[
    [np.ndarray[np.int8, 3]],
    tuple[
        np.ndarray[np.float64, 2],
        np.ndarray[np.float64, 2],
    ]
]
