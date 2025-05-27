# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from enum import Enum


class ArgFormat(str, Enum):
    """Tool argument formats"""

    XML = "xml"
    JSON = "json"
