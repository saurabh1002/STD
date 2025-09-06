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
from typing import Optional

import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.kiss_icp import KissICP
from kiss_icp.voxelization import voxel_down_sample

from stddesc.stddesc import STDesc
from stddesc.config import load_config
from stddesc.tools.pipeline_results import PipelineResults
from stddesc.tools.progress_bar import get_progress_bar


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t

def scan_to_map(scan_query, scan_ref, local_maps_scan_range):
    map_query = np.where((scan_query >= local_maps_scan_range[:, 0]) & (scan_query < local_maps_scan_range[:, 1]))[0][0]
    map_ref = np.where((scan_ref >= local_maps_scan_range[:, 0]) & (scan_ref < local_maps_scan_range[:, 1]))[0][0]
    return map_query, map_ref

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

        self._kiss_config = KISSConfig()
        self._kiss_config.mapping.voxel_size = self._kiss_config.data.max_range / 100.0
        self._odometry = KissICP(self._kiss_config)

        self.results_dir = results_dir
        self.config = load_config(config)
        self.std_desc = STDesc(self.config)

        self.closures = []

        self.dataset_name = self._dataset.sequence_id

        self.gt_closure_indices = self._dataset.gt_closure_indices
        self.local_maps_scan_range = self._dataset.local_maps_scan_range

        self.results = PipelineResults(self.gt_closure_indices)

    def run(self):
        self._run_pipeline()
        if self.gt_closure_indices is not None:
            self._run_evaluation()
        self._log_to_file()
        self._save_data()

        return self.results

    def _run_pipeline(self):
        start_pose_flag = True
        start_pose = np.eye(4)
        temp_cloud = []
        map_query = 0

        for i in get_progress_bar(self._first, self._last):
            try:
                frame, timestamps = self._dataset[i]
            except ValueError:
                frame = self._dataset[i]
                timestamps = np.array([])

            frame, _ = self._odometry.register_frame(frame, timestamps)
            pose = self._odometry.last_pose
            if start_pose_flag:
                start_pose = np.copy(pose)
                start_pose_flag = False
            frame_downsample = voxel_down_sample(frame, self.config.ds_size)
            delta_map_odom = np.linalg.inv(start_pose) @ pose
            temp_cloud.append(transform_points(frame_downsample, delta_map_odom))
            if np.linalg.norm(delta_map_odom[:3, -1]) > 100.0 or (i == self._last - 1):
                local_map = np.concatenate(temp_cloud)
                num_matches = self.std_desc.process_new_scan(local_map)
                for match_idx in range(num_matches):
                    map_ref, score, relative_tf = self.std_desc.get_closure_data(match_idx)
                    if score > 0.6:
                        self.closures.append(
                            np.r_[
                                map_ref,
                                map_query,
                                np.linalg.inv(relative_tf).flatten(),
                            ]
                        )
                    if map_query - map_ref > 3:
                        self.results.append(map_ref, map_query, score)

                temp_cloud.clear()
                start_pose_flag = True
                map_query += 1

    def _run_evaluation(self):
        self.results.compute_metrics()

    def _log_to_file(self):
        self.results_dir = self._create_results_dir()
        if self.gt_closure_indices is not None:
            self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))

    def _save_data(self):
        np.savetxt(os.path.join(self.results_dir, "closures.txt"), np.asarray(self.closures))

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
