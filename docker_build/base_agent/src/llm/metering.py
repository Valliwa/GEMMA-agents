# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import DefaultDict
from collections import defaultdict

from ..types.llm_types import TokenUsage, Model

# A mapping from models to token usage and dollar cost
token_meter: DefaultDict[Model, TokenUsage] = defaultdict(TokenUsage)
budget_info: dict[str, None | int | float] = dict(
    start_time=None,  # start timestamp
    cost_budget=None,  # cost budget in USD
    time_budget=None,  # time budget in seconds
)


def get_total_cost() -> float:
    total = 0.0
    for model in Model:
        total += token_meter[model].calculate_cost(model.token_cost)
    return total


def get_total_usage() -> TokenUsage:
    usage = TokenUsage()
    for model in Model:
        usage += token_meter[model]
    return usage


class CallCounter:
    def __init__(self):
        self.count = 0

    def count_new_call(self):
        self.count += 1

    def get_count(self) -> int:
        return self.count


llm_call_counter = CallCounter()
