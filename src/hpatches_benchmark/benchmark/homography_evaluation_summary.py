from __future__ import annotations
from dataclasses import dataclass
from hpatches_benchmark.benchmark.homography_evaluation_iou import HomographyEvaluationIOU
from hpatches_benchmark.benchmark.tabular import Tabular
import numpy as np

__all__ = ['HomographyEvaluationIOUSummary']

@dataclass
class HomographyEvaluationIOUSummary(Tabular):
    """
    Summary of individual IOU homography evaluations.
    :param evals: The evaluations the summary is based on.
    :param auc: The area under the curve with percentage of correct homographies on the y-axis
        and IOU threshold [0-1] on the x-axis.
    """
    evals: list[HomographyEvaluationIOU]
    auc: float

    @staticmethod
    def construct(evaluations: list[HomographyEvaluationIOU]) -> HomographyEvaluationIOUSummary:
        """
        Constructs the summary from a list of evaluations.
        :param evaluations: The individual evaluation results. Assumes epsilon is the same for
            all of them.
        :returns: The summary.
        """
        all_imgs = np.stack([
            e.correct_homographies for e in evaluations
        ], axis=1).T
        eps = evaluations[0].epsilon # assume all epislons are the same
        percentage_correct = np.sum(all_imgs, axis=0) / all_imgs.shape[0]
        eps_pairs = np.stack([
            eps[:-1], eps[1:]
        ], axis=-1)
        perc_correct_pairs = np.stack([
            percentage_correct[:-1], percentage_correct[1:]
        ], axis=-1)
        # use trapezoidal area
        A = np.diff(eps_pairs, axis=-1).squeeze(-1)
        B = np.max(perc_correct_pairs, axis=-1)
        C = np.abs(np.diff(perc_correct_pairs, axis=-1)).squeeze(-1)
        auc = np.sum(A * (B - 0.5 * C))
        return HomographyEvaluationIOUSummary(
            evaluations, auc
        ) 

    @property
    def table_headings(self):
        return ['Correct Homographies (IOU) AUC']

    @property
    def table_body(self):
        return [self.auc]
