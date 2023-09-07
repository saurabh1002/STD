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


class LocalMapPair:
    def __init__(
        self,
        ref_indices: np.ndarray,
        query_indices: np.ndarray,
        ref_scan_poses: np.ndarray,
        query_scan_poses: np.ndarray,
        relative_tf: np.ndarray,
        closure_distance_thresholds: np.ndarray,
    ):
        self._ref_indices = ref_indices
        self._query_indices = query_indices
        self._ref_scan_poses = ref_scan_poses
        self._query_scan_poses = query_scan_poses
        self._relative_tf = relative_tf
        self._closure_distance_thresholds = closure_distance_thresholds
        self._scan_level_closures = self._compute_scan_level_closures()

    def _compute_scan_level_closures(self) -> Dict[int, Set[Tuple]]:
        # bring all poses to a common frame at the query map
        query_locs = (self._relative_tf @ self._query_scan_poses)[:, :3, -1].squeeze()
        ref_locs = (self._ref_scan_poses)[:, :3, -1].squeeze()

        scan_closure_pairs: Dict[int, List[Tuple]] = {}
        for t in self._closure_distance_thresholds:
            scan_closure_pairs[t] = []
        query_id_start = self._query_indices[0]
        ref_id_start = self._ref_indices[0]
        qq, rr = np.meshgrid(self._query_indices, self._ref_indices)
        distances = norm(query_locs[qq - query_id_start] - ref_locs[rr - ref_id_start], axis=2)
        for dist_key in self._closure_distance_thresholds:
            ids = np.where(distances < dist_key)
            for r_id, q_id in zip(ids[0] + ref_id_start, ids[1] + query_id_start):
                scan_closure_pairs[dist_key].append((r_id, q_id))
        for key in scan_closure_pairs.keys():
            scan_closure_pairs[key] = set(map(lambda x: tuple(sorted(x)), scan_closure_pairs[key]))
        return scan_closure_pairs


class PipelineResults:
    def __init__(
        self, gt_closures: np.ndarray, dataset_name: str, closure_distance_thresholds: np.ndarray
    ):
        self._dataset_name = dataset_name
        self._closure_distance_thresholds = closure_distance_thresholds

        self.closure_indices_list: List[Dict[int, Set[Tuple]]] = []
        self.scores_list: List = []

        self.predicted_closures: Dict[int, Dict[float, Set[Tuple[int]]]] = {}
        self.metrics: Dict[int, Dict[float, Metrics]] = {}

        gt_closures = gt_closures if gt_closures.shape[1] == 2 else gt_closures.T
        self.gt_closures: Set[Tuple[int]] = set(map(lambda x: tuple(sorted(x)), gt_closures))

    def print(self):
        self.log_to_console()

    def append(self, local_map_pair: Dict[int, Set[Tuple]], score: float):
        self.closure_indices_list.append(local_map_pair)
        self.scores_list.append(score)

    def compute_closures_and_metrics(
        self,
    ):
        for dist_key in self._closure_distance_thresholds:
            metrics_dict = {}
            closures_dict = {}
            for score_threshold in np.arange(0.1, 1.0, 0.05):
                closures = set()
                for closure_indices, score in zip(self.closure_indices_list, self.scores_list):
                    if score >= score_threshold:
                        closures = closures.union(closure_indices[dist_key])
                closures_dict[score_threshold] = closures

                tp = len(self.gt_closures.intersection(closures))
                fp = len(closures) - tp
                fn = len(self.gt_closures) - tp
                metrics_dict[score_threshold] = Metrics(tp, fp, fn)
            self.predicted_closures[dist_key] = closures_dict
            self.metrics[dist_key] = metrics_dict

    def _rich_table_pr(self, table_format: box.Box = box.HORIZONTALS) -> Set[Table]:
        tables = []
        for dist_key in self._closure_distance_thresholds:
            table = Table(box=table_format, title=self._dataset_name)
            table.caption = f"Loop Closure Distance Threshold: {dist_key}m"
            table.add_column("Score Threshold", justify="center", style="cyan")
            table.add_column("True Positives", justify="center", style="magenta")
            table.add_column("False Positives", justify="center", style="magenta")
            table.add_column("False Negatives", justify="center", style="magenta")
            table.add_column("Precision", justify="left", style="green")
            table.add_column("Recall", justify="left", style="green")
            table.add_column("F1 score", justify="left", style="green")
            for [threshold, metric] in self.metrics[dist_key].items():
                table.add_row(
                    f"{threshold:.4f}",
                    f"{metric.tp}",
                    f"{metric.fp}",
                    f"{metric.fn}",
                    f"{metric.precision:.4f}",
                    f"{metric.recall:.4f}",
                    f"{metric.F1:.4f}",
                )
            tables.append(table)
        return tables

    def log_to_console(self):
        console = Console()
        for table in self._rich_table_pr():
            console.print(table)

    def log_to_file_pr(self, filename):
        with open(filename, "wt") as logfile:
            console = Console(file=logfile, width=100, force_jupyter=False)
            for table in self._rich_table_pr(table_format=box.ASCII_DOUBLE_HEAD):
                console.print(table)

    def log_to_file_closures(self, result_dir):
        np.save(
            os.path.join(result_dir, f"predicted_closures.npy"),
            self.predicted_closures,
        )
