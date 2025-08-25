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
from kiss_icp.voxelization import voxel_down_sample

from stddesc.stddesc import STDesc
from stddesc.config import load_config
from stddesc.tools.pipeline_results import PipelineResults
from stddesc.tools.progress_bar import get_progress_bar


def transform_points(pcd, T):
    R = T[:3, :3]
    t = T[:3, -1]
    return pcd @ R.T + t


def scan_to_map(scan_query, scan_ref, query_local_maps_scan_range, ref_local_maps_scan_range):
    map_query = np.where(
        (scan_query >= query_local_maps_scan_range[:, 0])
        & (scan_query < query_local_maps_scan_range[:, 1])
    )[0][0]
    map_ref = np.where(
        (scan_ref >= ref_local_maps_scan_range[:, 0]) & (scan_ref < ref_local_maps_scan_range[:, 1])
    )[0][0]
    return map_query, map_ref


class STDescPipeline:
    def __init__(
        self,
        dataset_query,
        dataset_ref,
        results_dir,
        config: Optional[Path] = None,
    ):
        self._query_dataset = dataset_query
        self._query_dataset_name = (
            self._query_dataset.sequence_id
            if hasattr(self._query_dataset, "sequence_id")
            else os.path.basename(self._query_dataset.data_dir)
        )
        self._ref_dataset = dataset_ref
        self._ref_dataset_name = (
            self._ref_dataset.sequence_id
            if hasattr(self._ref_dataset, "sequence_id")
            else os.path.basename(self._ref_dataset.data_dir)
        )

        self.results_dir = results_dir
        self.config = load_config(config)
        self.std_desc = STDesc(self.config)

        self.ref_map_scan_indices = []
        self.query_map_scan_indices = []
        self.closures = []

        base_dir_query = self._query_dataset.sequence_dir
        file_path_closures_query = os.path.join(
            base_dir_query,
            "loop_closure",
            f"{self._ref_dataset.sequence_id}_local_map_gt_closures.txt",
        )
        if os.path.exists(file_path_closures_query) and os.path.exists(file_path_closures_query):
            self.gt_closures = np.loadtxt(file_path_closures_query, dtype=int)
            print(f"[INFO] Found closure ground truth at {file_path_closures_query}")
        else:
            self.gt_closures = None
            print(f"[INFO] No closure ground truth found at {file_path_closures_query}")

        self.ref_local_maps_scan_range = self._ref_dataset.local_maps_scan_range
        self.query_local_maps_scan_range = self._query_dataset.local_maps_scan_range

        self.results = PipelineResults(self.gt_closures)

    def run(self):
        self._run_pipeline()
        self._run_evaluation()
        self._log_to_file()
        self._save_data()

        return self.results

    def _run_pipeline(self):
        start_pose_flag = True
        start_pose = np.eye(4)
        temp_cloud = []
        ref_scan_indices = []

        for i in get_progress_bar(0, len(self._ref_dataset)):
            try:
                frame, _ = self._ref_dataset[i]
            except ValueError:
                frame = self._ref_dataset[i]

            pose = self._ref_dataset.kiss_poses[i]
            if start_pose_flag:
                start_pose = np.copy(pose)
                start_pose_flag = False
            frame_downsample = voxel_down_sample(frame, self.config.ds_size)
            delta_map_odom = np.linalg.inv(start_pose) @ pose
            temp_cloud.append(transform_points(frame_downsample, delta_map_odom))
            if ((i + 1) % self.config.sub_frame_num) == 0:
                ref_scan_indices.append(i)
                self.ref_map_scan_indices.append(np.array(ref_scan_indices))

                local_map = np.concatenate(temp_cloud)
                self.std_desc.add_to_database(local_map)

                temp_cloud.clear()
                ref_scan_indices.clear()
                start_pose_flag = True
            else:
                ref_scan_indices.append(i)

        start_pose_flag = True
        start_pose = np.eye(4)
        query_scan_indices = []
        query_idx = 0

        for i in get_progress_bar(0, len(self._query_dataset)):
            try:
                frame, _ = self._query_dataset[i]
            except ValueError:
                frame = self._query_dataset[i]

            pose = self._query_dataset.kiss_poses[i]
            if start_pose_flag:
                start_pose = np.copy(pose)
                start_pose_flag = False
            frame_downsample = voxel_down_sample(frame, self.config.ds_size)
            delta_map_odom = np.linalg.inv(start_pose) @ pose
            temp_cloud.append(transform_points(frame_downsample, delta_map_odom))
            if ((i + 1) % self.config.sub_frame_num) == 0:
                query_scan_indices.append(i)
                self.query_map_scan_indices.append(np.array(query_scan_indices))

                local_map = np.concatenate(temp_cloud)
                num_matches = self.std_desc.compute_closures(local_map)
                for match_idx in range(num_matches):
                    ref_idx, score, relative_tf = self.std_desc.get_closure_data(match_idx)
                    if score > 0.6:
                        self.closures.append(
                            np.r_[
                                ref_idx,
                                query_idx,
                                np.linalg.inv(relative_tf).flatten(),
                            ]
                        )
                    for ref_id in self.ref_map_scan_indices[ref_idx]:
                        for query_id in self.query_map_scan_indices[query_idx]:
                            map_query, map_ref = scan_to_map(
                                query_id,
                                ref_id,
                                self.query_local_maps_scan_range,
                                self.ref_local_maps_scan_range,
                            )
                            self.results.append(map_ref, map_query, score)

                temp_cloud.clear()
                query_scan_indices.clear()
                start_pose_flag = True
                query_idx += 1
            else:
                query_scan_indices.append(i)

    def _run_evaluation(self):
        self.results.compute_metrics()

    def _log_to_file(self):
        self.results_dir = self._create_results_dir()
        self.results.log_to_file_pr(os.path.join(self.results_dir, "metrics.txt"))

    def _save_data(self):
        np.savetxt(os.path.join(self.results_dir, "multi-session_closures.txt"), np.asarray(self.closures))

    def _create_results_dir(self) -> Path:
        results_dir = os.path.join(
            self.results_dir, f"{self._query_dataset_name}", f"{self._ref_dataset_name}"
        )
        os.makedirs(results_dir, exist_ok=True)

        return results_dir
