from __future__ import annotations
from dataclasses import dataclass
from hpatches_benchmark.benchmark.evaluation_step import EvaluationStep
import numpy as np
from tabulate import tabulate
from itertools import compress

def _compute_homo(eval_step: list[EvaluationStep], eps: float) -> float:
    loc_distances = np.mean(np.array([
        x.homography_metrics.img_corner_distances for x in eval_step
    ]), axis=-1)
    n_viewpoint = loc_distances.shape[0]
    metric = np.count_nonzero(loc_distances < eps) / n_viewpoint
    return metric

def _compute_mle(eval_step: list[EvaluationStep], eps: float) -> float:
    loc_distances = np.mean(np.array([
        x.homography_metrics.img_corner_distances for x in eval_step
    ]), axis=-1)
    mle = np.mean(
        np.array(
            [dist for x in compress(eval_step, loc_distances < eps)
                for dist in x.homography_metrics.kp_distances]
        )
    )
    return mle

@dataclass
class BenchmarkResult:
    viewpoint_eval_steps: list[EvaluationStep]
    intensity_eval_steps: list[EvaluationStep]
    n_viewpoint: int
    homo_eps1: float
    homo_eps3: float
    homo_eps5: float
    mle_eps1: float
    mle_eps3: float
    mle_eps5: float

    @staticmethod
    def compute(viewpoint_steps: list[EvaluationStep],
                intensity_steps: list[EvaluationStep]) -> BenchmarkResult:
        eps1_correct = _compute_homo(viewpoint_steps, 1.0)
        eps3_correct = _compute_homo(viewpoint_steps, 3.0)
        eps5_correct = _compute_homo(viewpoint_steps, 5.0)
        mle_eps1 = _compute_mle(viewpoint_steps, 1.0)
        mle_eps3 = _compute_mle(viewpoint_steps, 3.0)
        mle_eps5 = _compute_mle(viewpoint_steps, 5.0)
        benchmark_result = BenchmarkResult(viewpoint_steps, intensity_steps, len(viewpoint_steps),
                               eps1_correct, eps3_correct, eps5_correct, mle_eps1,
                               mle_eps3, mle_eps5)
        return benchmark_result

    def table(self) -> str:
        return tabulate(
            [[self.homo_eps1, self.homo_eps3, self.homo_eps5, self.mle_eps1,
              self.mle_eps3, self.mle_eps5, self.n_viewpoint]],
            headers=['HOMO EPS 1', 'HOMO EPS 3', 'HOMO EPS 5', 'MLE EPS 1',
                     'MLE EPS3', 'MLE EPS5', 'Num Viewpoint Images']
        )