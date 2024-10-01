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

from pybind.stdesc import STDesc
from stdesc.config import load_config
from stdesc.tools.pipeline_results import LocalMapPair, PipelineResults
from stdesc.tools.progress_bar import get_progress_bar


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


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

        self.map_scan_indices = []
        self.map_scan_poses = []
        self.closures = []

        self.dataset_name = self._dataset.sequence_id

        self.gt_closure_indices = self._dataset.gt_closure_indices

        self.closure_distance_thresholds = np.arange(1, 10, 0.5)
        self.results = PipelineResults(
            self.gt_closure_indices, self.dataset_name, self.closure_distance_thresholds
        )

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
        query_scan_indices = []
        query_scan_poses = []
        query_idx = 0

        for i in get_progress_bar(self._first, self._last):
            scan = self._dataset[i]
            self._odometry.register_frame(scan, 0)
            pose = self._odometry.last_pose
            if start_pose_flag:
                start_pose = np.copy(pose)
                start_pose_flag = False
            frame_downsample = voxel_down_sample(scan, self.config.ds_size)
            delta_map_odom = np.linalg.inv(start_pose) @ pose
            temp_cloud.append(transform_points(frame_downsample, delta_map_odom))
            if ((i + 1) % self.config.sub_frame_num) == 0:
                query_scan_indices.append(i)
                query_scan_poses.append(pose)
                self.map_scan_indices.append(np.array(query_scan_indices))
                self.map_scan_poses.append(np.array(query_scan_poses))

                local_map = np.concatenate(temp_cloud)
                num_matches = self.std_desc.process_new_scan(local_map)
                for match_idx in range(num_matches):
                    ref_idx, score, relative_tf = self.std_desc.get_closure_data(match_idx)
                    if score > 0.6:
                        self.closures.append(
                            np.r_[
                                ref_idx,
                                query_idx,
                                self.map_scan_indices[ref_idx][0],
                                self.map_scan_indices[query_idx][0],
                                np.linalg.inv(relative_tf).flatten(),
                            ]
                        )
                    local_map_pairs = LocalMapPair(
                        self.map_scan_indices[ref_idx],
                        self.map_scan_indices[query_idx],
                        self.map_scan_poses[ref_idx],
                        self.map_scan_poses[query_idx],
                        relative_tf,
                        self.closure_distance_thresholds,
                    )
                    self.results.append(local_map_pairs._scan_level_closures, score)
                temp_cloud.clear()
                query_scan_indices.clear()
                query_scan_poses.clear()
                start_pose_flag = True
                query_idx += 1
            else:
                query_scan_indices.append(i)
                query_scan_poses.append(pose)

    def _run_evaluation(self):
        self.results.compute_closures_and_metrics()

    def _log_to_file(self):
        self.results_dir = self._create_results_dir()
        if self.gt_closure_indices is not None:
            self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))
        self.results.log_to_file_closures(self.results_dir)

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
