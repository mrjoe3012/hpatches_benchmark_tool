from __future__ import annotations
from dataclasses import dataclass
from ndshapecheck import ShapeCheck
import numpy as np
from hpatches_benchmark.benchmark.features import Features
from hpatches_benchmark.benchmark.tabular import Tabular

__all__ = ['RepeatabilityEvaluation']

@dataclass
class RepeatabilityEvaluation(Tabular):
    """
    Evaluation of the repeatability of features.
    :param features:
    :param epsilon: Correctness thresholds.
    :param repeatability:
    """
    features: Features
    epsilon: np.ndarray[np.float64, 1]
    repeatability: np.ndarray[np.float64, 1]
    n_kp: int

    def __post_init__(self) -> None:
        sc = ShapeCheck()
        assert sc('N').check(self.epsilon.shape), sc.why
        assert sc('N').check(self.repeatability.shape), sc.why

    @property
    def table_headings(self):
        return [
            f'Rep @ {eps:.2f}' for eps in self.epsilon
        ]

    @property
    def table_body(self):
        return self.repeatability.tolist()

    @staticmethod
    def construct_empty(features: Features, epsilon: np.ndarray[np.float64, 1],
                        n_kp: int) -> RepeatabilityEvaluation:
        return RepeatabilityEvaluation(
            features, epsilon,
            np.full(epsilon.shape, 0.0), n_kp
        )
