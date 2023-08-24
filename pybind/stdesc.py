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

from typing import Tuple

import numpy as np
from pydantic import BaseSettings

from . import stdesc_pybind


class STDesc:
    def __init__(self, config: BaseSettings):
        self._config = config
        self._pipeline = stdesc_pybind._STDescManager(self._config.dict())

    def process_new_scan(self, scan: np.ndarray) -> Tuple[int, float]:
        scan = stdesc_pybind._VectorEigen3d(scan)
        num_matches = self._pipeline._ProcessNewScan(scan)
        return num_matches

    def get_closure_data(self, idx: int) -> Tuple[int, float, np.ndarray]:
        match_idx, match_score, t, R = self._pipeline._GetClosureDataAtIdx(idx)
        T = np.eye(4)
        T[:3, :3] = np.asarray(R)
        T[:3, -1] = np.asarray(t)
        return match_idx, match_score, T


def voxel_down_sample(scan: np.ndarray, voxel_size: float) -> np.ndarray:
    scan = stdesc_pybind._VectorEigen3d(scan)
    stdesc_pybind._VoxelDownSample(scan, voxel_size)
    return np.asarray(scan)
