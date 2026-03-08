from __future__ import annotations
from dataclasses import dataclass

from hpatches_benchmark.benchmark.homography_evaluation import HomographyEvaluation
from hpatches_benchmark.benchmark.homography_evaluation_iou import HomographyEvaluationIOU
from hpatches_benchmark.benchmark.homography_evaluation_summary import HomographyEvaluationIOUSummary
from hpatches_benchmark.benchmark.matching_score_evaluation import MatchingScoreEvaluation
from hpatches_benchmark.benchmark.mma_evaluation import MMAEvaluation
from hpatches_benchmark.benchmark.repeatability_evaluation import RepeatabilityEvaluation
from hpatches_benchmark.benchmark.tabular import Tabular
from hpatches_benchmark.dataset.hpatches import HPatches
import pandas as pd
import numpy as np
import re

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
    homography_evaluation_iou: list[HomographyEvaluationIOU]
    repeatability_evaluation: list[RepeatabilityEvaluation]
    mma_evaluation: list[MMAEvaluation]
    m_score_evaluation: list[MatchingScoreEvaluation]
    homography_evaluation_iou_summary: HomographyEvaluationIOUSummary = None

    def __post_init__(self) -> None:
        assert len(self.homography_evaluation) == len(self.repeatability_evaluation) \
            == len(self.mma_evaluation) == len(self.m_score_evaluation) == len(self.homography_evaluation_iou)
        self.homography_evaluation_iou_summary = HomographyEvaluationIOUSummary.construct(
            self.homography_evaluation_iou
        )

    def split_by_task(self) -> tuple[BenchmarkResult, BenchmarkResult]:
        """
        :returns: (benchmark of intensity images, benchmark of viewpoint images)
        """
        intensity_homo, vp_homo  = [], []
        intensity_homo_iou, vp_homo_iou  = [], []
        for eval1, eval2 in zip(self.homography_evaluation, self.homography_evaluation_iou):
            task = eval1.homography_estimate.matches.features.img.task
            if task == 'intensity':
                intensity_homo.append(eval1)
                intensity_homo_iou.append(eval2)
            else:
                vp_homo.append(eval1)
                vp_homo_iou.append(eval2)
        intensity_rep, vp_rep = [], []
        for eval in self.repeatability_evaluation:
            if eval.features.img.task == 'intensity':
                intensity_rep.append(eval)
            else:
                vp_rep.append(eval)
        intensity_mma, vp_mma = [], []
        for eval in self.mma_evaluation:
            if eval.matches.features.img.task == 'intensity':
                intensity_mma.append(eval)
            else:
                vp_mma.append(eval)
        intensity_m_score, vp_m_score = [], []
        for eval in self.m_score_evaluation:
            if eval.matches.features.img.task == 'intensity':
                intensity_m_score.append(eval)
            else:
                vp_m_score.append(eval)
        return (
            BenchmarkResult(self.hpatches, intensity_homo, intensity_homo_iou, intensity_rep, intensity_mma, intensity_m_score),
            BenchmarkResult(self.hpatches, vp_homo, vp_homo_iou, vp_rep, vp_mma, vp_m_score)
        )

    @property
    def dataframe(self) -> pd.DataFrame:
        headings = self.homography_evaluation[0].table_headings \
            + self.repeatability_evaluation[0].table_headings \
            + self.mma_evaluation[0].table_headings \
            + self.m_score_evaluation[0].table_headings \
            + self.homography_evaluation_iou[0].table_headings \
            + self.homography_evaluation_iou_summary.table_headings
        body = [
            homo.table_body + rep.table_body + mma.table_body + ms.table_body \
                + iou.table_body + self.homography_evaluation_iou_summary.table_body
                    for homo, rep, mma, ms, iou in zip(self.homography_evaluation,
                                            self.repeatability_evaluation,
                                            self.mma_evaluation,
                                            self.m_score_evaluation,
                                            self.homography_evaluation_iou,
                                            strict=True)
        ]
        return pd.DataFrame(data=body, columns=headings)

    @property
    def summary_dataframe(self) -> pd.DataFrame:
        full_df = self.dataframe
        homo_df = full_df[[col for col in full_df.columns if 'Correct Homo' in col and 'IOU' not in col]]
        homo_stats = homo_df.mean()
        homo_stderr = pd.Series([homo_stats.std()], ['Correct Homo Std Err'])
        mle_df = full_df[[col for col in full_df.columns if 'MLE' in col]]
        n_pts = full_df['Num Points'].to_numpy()
        mle_sum = mle_df.mul(n_pts[:, None])
        mle_stats = mle_sum.sum() / np.sum(n_pts)
        mle_stderr = pd.Series([mle_stats.std()], ['MLE Std Err'])
        repeatability_stats = full_df[[col for col in full_df.columns if 'Rep' in col]].mean()
        repeatability_stderr = pd.Series([repeatability_stats.std()], ['Repeatability Std Err'])
        mma_stats = full_df[[col for col in full_df.columns if 'MMA' in col]].mean()
        mma_stderr = pd.Series([mma_stats.std()], ['MMA Std Err'])
        ms_stats = full_df[[col for col in full_df.columns if 'M SCORE' in col]].mean()
        ms_stderr = pd.Series([ms_stats.std()], ['M SCORE Std Err'])
        homo_iou_stats = full_df[[col for col in full_df.columns if 'IOU' in col]].mean()
        joined = pd.concat([
            homo_stats, homo_stderr,
            homo_iou_stats,
            mle_stats, mle_stderr,
            repeatability_stats, repeatability_stderr,
            mma_stats, mma_stderr,
            ms_stats, ms_stderr
        ], axis=0).to_frame().T
        # trim down the epsilon thresholds to only 1, 3, 5
        joined = joined[[
            col for col in joined.columns 
                if '@' not in col or re.search(r'@ [135]\.00', col) is not None
        ]]
        return joined
