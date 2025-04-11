# MIT License
#
# Copyright (c) 2023 Saurabh Gupta, Tiziano Guadagnino, Cyrill Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
from typing import Dict, List, Set, Tuple

import numpy as np
from numpy.linalg import inv, norm
from rich import box
from rich.console import Console
from rich.table import Table


class Metrics:
    def __init__(self, true_positives, false_positives, false_negatives):
        self.tp = true_positives
        self.fp = false_positives
        self.fn = false_negatives

        try:
            self.precision = self.tp / (self.tp + self.fp)
        except ZeroDivisionError:
            self.precision = np.nan

        try:
            self.recall = self.tp / (self.tp + self.fn)
        except ZeroDivisionError:
            self.recall = np.nan

        try:
            self.F1 = 2 * self.precision * self.recall / (self.precision + self.recall)
        except ZeroDivisionError:
            self.F1 = np.nan

    def __call__(self):
        return np.r_[self.tp, self.fp, self.fn, self.precision, self.recall, self.F1]


def compute_closure_indices(
    ref_indices: np.ndarray,
    query_indices: np.ndarray,
    ref_scan_poses: np.ndarray,
    query_scan_poses: np.ndarray,
    relative_tf: np.ndarray,
    closure_distance_threshold: float,
):
    T_query_world = inv(query_scan_poses[0])
    T_ref_world = inv(ref_scan_poses[0])

    # bring all poses to a common frame at the query map
    query_locs = (relative_tf @ T_query_world @ query_scan_poses)[:, :3, -1].squeeze()
    ref_locs = (T_ref_world @ ref_scan_poses)[:, :3, -1].squeeze()

    closure_indices = []
    closure_distances = []
    query_id_start = query_indices[0]
    ref_id_start = ref_indices[0]
    qq, rr = np.meshgrid(query_indices, ref_indices)
    distances = norm(query_locs[qq - query_id_start] - ref_locs[rr - ref_id_start], axis=2)
    ids = np.where(distances < closure_distance_threshold)
    for r_id, q_id, distance in zip(ids[0] + ref_id_start, ids[1] + query_id_start, distances[ids]):
        closure_indices.append((r_id, q_id))
        closure_distances.append(distance)
    return np.asarray(closure_indices, int), np.asarray(closure_distances)


class PipelineResults:
    def __init__(
        self, gt_closures: np.ndarray, dataset_name: str, closure_distance_threshold: float
    ):
        self._dataset_name = dataset_name
        self._closure_distance_threshold = closure_distance_threshold

        self.closure_indices_list: List[List[Tuple[int]]] = []
        self.closure_distances_list: List[List[float]] = []
        self.scores_list: List = []

        self.metrics = np.zeros((18, closure_distance_threshold - 1, 6))

        gt_closures = gt_closures if gt_closures.shape[1] == 2 else gt_closures.T
        self.gt_closures: Set[Tuple[int]] = set(map(lambda x: tuple(sorted(x)), gt_closures))

    def print(self):
        self.log_to_console()

    def append(
        self,
        ref_indices: np.ndarray,
        query_indices: np.ndarray,
        ref_scan_poses: np.ndarray,
        query_scan_poses: np.ndarray,
        relative_tf: np.ndarray,
        closure_distance_threshold: float,
        score: float,
    ):
        closure_indices, closure_distances = compute_closure_indices(
            ref_indices,
            query_indices,
            ref_scan_poses,
            query_scan_poses,
            relative_tf,
            closure_distance_threshold,
        )
        self.closure_indices_list.append(closure_indices)
        self.closure_distances_list.append(closure_distances)
        self.scores_list.append(score)

    def compute_closures_and_metrics(
        self,
    ):
        for i, score_threshold in enumerate(np.arange(0.1, 1.0, 0.05)):
            for j, distance_threshold in enumerate(range(1, self._closure_distance_threshold)):
                closures = set()
                for closure_indices, closure_distances, score in zip(
                    self.closure_indices_list, self.closure_distances_list, self.scores_list
                ):
                    if score >= score_threshold:
                        closures = closures.union(
                            set(
                                map(
                                    lambda x: tuple(x),
                                    closure_indices[
                                        np.where(closure_distances < distance_threshold)
                                    ],
                                )
                            )
                        )

                tp = len(self.gt_closures.intersection(closures))
                fp = len(closures) - tp
                fn = len(self.gt_closures) - tp
                self.metrics[i, j] = Metrics(tp, fp, fn)()

    def _rich_table_pr(self, table_format: box.Box = box.HORIZONTALS) -> Table:
        table = Table(box=table_format)
        table.caption = f"Loop Closure Evaluation Metrics\n"
        table.add_column("Score \ Distance (m)", justify="center", style="cyan")
        score_thresholds = np.arange(0.1, 1.0, 0.05)
        for distance_threshold in range(1, self._closure_distance_threshold):
            table.add_column(f"{distance_threshold}", justify="center", style="magenta")
        for i, row in enumerate(self.metrics):
            metrics = [f"{val[-3]:.4f}\n{val[-2]:.4f}\n{val[-1]:.4f}" for val in row]
            table.add_row(f"{score_thresholds[i]:.2f}", *metrics)
        return table

    def log_to_console(self):
        console = Console()
        table = self._rich_table_pr()
        console.print(table)

    def log_to_file_pr(self, filename):
        with open(filename, "wt") as logfile:
            console = Console(file=logfile, width=100, force_jupyter=False)
            table = self._rich_table_pr(table_format=box.ASCII_DOUBLE_HEAD)
            console.print(table)
