from __future__ import annotations
from dataclasses import dataclass

from hpatches_benchmark.benchmark.homography_evaluation import HomographyEvaluation
from hpatches_benchmark.benchmark.repeatability_evaluation import RepeatabilityEvaluation
from hpatches_benchmark.benchmark.tabular import Tabular
from hpatches_benchmark.dataset.hpatches import HPatches
import pandas as pd
import numpy as np

@dataclass
class BenchmarkResult:
    """
    Results from running the full benchmark.
    :param hpatches: The dataset.
    :param homography_evaluation:
    :param repeatability_evaluation: 
    """
    hpatches: HPatches
    homography_evaluation: list[HomographyEvaluation]
    repeatability_evlauation: list[RepeatabilityEvaluation]

    def __post_init__(self) -> None:
        assert len(self.homography_evaluation) == len(self.repeatability_evlauation)

    def split_by_task(self) -> tuple[BenchmarkResult, BenchmarkResult]:
        """
        :returns: (benchmark of intensity images, benchmark of viewpoint images)
        """
        intensity_homo, vp_homo  = [], []
        for eval in self.homography_evaluation:
            if eval.homography_estimate.matches.features.img.task == 'intensity':
                intensity_homo.append(eval)
            else:
                vp_homo.append(eval)
        intensity_rep, vp_rep = [], []
        for eval in self.repeatability_evlauation:
            if eval.features.img.task == 'intensity':
                intensity_rep.append(eval)
            else:
                vp_rep.append(eval)
        return (
            BenchmarkResult(self.hpatches, intensity_homo, intensity_rep),
            BenchmarkResult(self.hpatches, vp_homo, vp_rep)
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        headings = self.homography_evaluation[0].table_headings \
            + self.repeatability_evlauation[0].table_headings
        body = [
            homo.table_body +  rep.table_body
                for homo, rep in zip(self.homography_evaluation,
                                     self.repeatability_evlauation,
                                     strict=True)
        ]
        return pd.DataFrame(data=body, columns=headings)

    @property
    def summary_dataframe(self) -> pd.DataFrame:
        full_df = self.dataframe
        homo_df = full_df[[col for col in full_df.columns if 'Correct Homo' in col]]
        homo_stats = homo_df.mean()
        mle_df = full_df[[col for col in full_df.columns if 'MLE' in col]]
        n_pts = full_df['Num Points'].to_numpy()
        mle_sum = mle_df.mul(n_pts[:, None])
        mle_stats = mle_sum.sum() / np.sum(n_pts)
        repeatability_stats = full_df[[col for col in full_df.columns if 'Rep' in col]].mean()
        joined = pd.concat([homo_stats, mle_stats, repeatability_stats], axis=0).to_frame().T
        return joined
