# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

from datetime import datetime, timedelta

from ..types.llm_types import TokenUsage
from ..types.agent_types import AgentMetrics


def make_random_agent_metrics(
    tools_enabled: bool = True,
    agents_enabled: bool = True,
    min_duration_seconds: int = 1,
    max_duration_seconds: int = 300,
    base_prompt_tokens: int = 500,
    token_variance: float = 0.3,
    cache_hit_rate: float = 0.4,
    cache_write_rate: float = 0.3,
    cost_per_1k_tokens: float = 0.002,
    seed: int = 42  # Added seed parameter
) -> AgentMetrics:
    """
    Generate random but plausible agent metrics deterministically.

    Args:
        tools_enabled: Whether tools are enabled for this agent
        agents_enabled: Whether sub-agents are enabled for this agent
        min_duration_seconds: Minimum execution duration in seconds
        max_duration_seconds: Maximum execution duration in seconds
        base_prompt_tokens: Base number of prompt tokens to vary around
        token_variance: How much to vary token counts (as proportion of base)
        cache_hit_rate: Proportion of tokens that should be cached hits
        cache_write_rate: Proportion of uncached tokens that should be written to cache
        cost_per_1k_tokens: Cost per 1000 tokens in dollars
        seed: Random seed for deterministic output

    Returns:
        AgentMetrics object with randomized but plausible values
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Use a fixed base time instead of datetime.now()
    base_time = datetime(2025, 1, 1, 0, 0, 0)  # Fixed starting point
    start_time = base_time - timedelta(days=random.randint(0, 7))
    duration = random.uniform(min_duration_seconds, max_duration_seconds)
    end_time = start_time + timedelta(seconds=duration)

    # Calculate base token counts with some variance
    variance_factor = 1 + random.uniform(-token_variance, token_variance)
    total_prompt_tokens = int(base_prompt_tokens * variance_factor)

    # Calculate cached vs uncached split
    cached_tokens = int(total_prompt_tokens * cache_hit_rate)
    uncached_tokens = total_prompt_tokens - cached_tokens

    # Calculate cache writes
    cache_writes = int(uncached_tokens * cache_write_rate)

    # Generate completion tokens (typically 20-80% of prompt tokens)
    completion_tokens = int(total_prompt_tokens * random.uniform(0.2, 0.8))

    # Calculate tool and agent calls if enabled
    tool_calls = 0
    agent_calls = 0

    if tools_enabled:
        # Typically 1-5 tool calls per interaction
        tool_calls = random.randint(1, 5)

    if agents_enabled:
        # Typically 0-3 agent calls per interaction
        agent_calls = random.randint(0, 3)

    # Calculate total cost
    total_tokens = total_prompt_tokens + completion_tokens
    cost = (total_tokens / 1000) * cost_per_1k_tokens

    return AgentMetrics(
        start_time=start_time,
        end_time=end_time,
        token_usage=TokenUsage(
            uncached_prompt_tokens=uncached_tokens - cache_writes,
            cache_write_prompt_tokens=cache_writes,
            cached_prompt_tokens=cached_tokens,
            completion_tokens=completion_tokens,
        ),
        cost=cost,
        tool_calls=tool_calls,
        agent_calls=agent_calls,
    )
