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
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseSettings, PrivateAttr


class STDescConfig(BaseSettings):
    ds_size: float = 0.25
    maximum_corner_num: int = 100
    plane_merge_normal_thre: float = 0.2
    plane_detection_thre: float = 0.01
    voxel_size: float = 2.0
    voxel_init_num: int = 10
    proj_image_resolution: float = 0.5
    proj_dis_min: float = 0
    proj_dis_max: float = 5
    corner_thre: float = 10
    descriptor_near_num: int = 10
    descriptor_min_len: float = 2
    descriptor_max_len: float = 50
    non_max_suppression_radius: float = 2
    std_side_resolution: float = 0.2
    skip_near_num: int = 50
    candidate_num: int = 50
    sub_frame_num: int = 10
    rough_dis_threshold: float = 0.01
    vertex_diff_threshold: float = 0.5
    icp_threshold: float = 0.4
    normal_threshold: float = 0.2
    dis_threshold: float = 0.5

    _config_file: Optional[Path] = PrivateAttr()

    def __init__(self, config_file: Optional[Path] = None, *args, **kwargs):
        self._config_file = config_file
        super().__init__(*args, **kwargs)

    def _yaml_source(self) -> Dict[str, Any]:
        data = None
        if self._config_file is not None:
            try:
                yaml = importlib.import_module("yaml")
            except ModuleNotFoundError:
                print(
                    "Custom configuration file specified but PyYAML is not installed on your system,"
                    " run `pip install pyyaml`"
                )
                sys.exit(1)
            with open(self._config_file) as cfg_file:
                data = yaml.safe_load(cfg_file)
        return data or {}

    class Config:
        @classmethod
        def customise_sources(cls, init_settings, env_settings, file_secret_settings):
            return init_settings, STDescConfig._yaml_source


def load_config(config_file: Optional[Path]) -> STDescConfig:
    """Load configuration from an Optional yaml file. Additionally, deskew and max_range can be
    also specified from the CLI interface"""

    config = STDescConfig(config_file=config_file)

    return config


def write_config(config: STDescConfig, filename: str):
    with open(filename, "w") as outfile:
        try:
            yaml = importlib.import_module("yaml")
            yaml.dump(config.dict(), outfile, default_flow_style=False)
        except ModuleNotFoundError:
            outfile.write(str(config.dict()))
