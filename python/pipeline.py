# MIT License
#
# Copyright (c) 2023 Saurabh Gupta, Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch,
# Cyrill Stachniss.
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
import datetime
import os
from pathlib import Path
from typing import NoReturn, Optional

import numpy as np

from pybind.stdesc import STDesc
from python.config import load_config
from python.tools.pipeline_results import PipelineResults
from python.tools.progress_bar import get_progress_bar


class STDescPipeline:
    def __init__(
        self,
        dataset,
        results_dir,
        config: Optional[Path] = None,
    ):
        self._dataset = dataset
        self._first = 0
        self._last = len(self._dataset)

        self.results_dir = results_dir
        self.config = load_config(config)
        self.std_desc = STDesc(self.config)

        self.dataset_name = self._dataset.sequence_id

        self.gt_closure_indices = self._dataset.gt_closure_indices

        stdesc_thresholds = np.arange(0.1, 1.0, 0.1)
        self.results = PipelineResults(
            self.gt_closure_indices, self.dataset_name, stdesc_thresholds
        )

    def run(self):
        self._run_pipeline()
        if self.gt_closure_indices is not None:
            self._run_evaluation()
        self._log_to_file()

        return self.results

    def _run_pipeline(self):
        for query_idx in get_progress_bar(self._first, self._last):
            scan = self._dataset[query_idx]
            closure_idx, score = self.std_desc.process_new_scan(scan, query_idx)
            if closure_idx != -1:
                self.results.append(query_idx, closure_idx, score)

    def _run_evaluation(self) -> NoReturn:
        self.results.compute_metrics()

    def _log_to_file(self) -> NoReturn:
        self.results_dir = self._create_results_dir()
        if self.gt_closure_indices is not None:
            self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))
        self.results.log_to_file_closures(self.results_dir)

    def _create_results_dir(self) -> Path:
        def get_timestamp() -> str:
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        results_dir = os.path.join(
            self.results_dir, "stdesc_results", self.dataset_name, get_timestamp()
        )
        latest_dir = os.path.join(self.results_dir, "stdesc_results", self.dataset_name, "latest")
        os.makedirs(results_dir, exist_ok=True)
        os.unlink(latest_dir) if os.path.exists(latest_dir) or os.path.islink(latest_dir) else None
        os.symlink(results_dir, latest_dir)

        return results_dir
