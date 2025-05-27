# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Built-in agent agents providing core capabilities."""

from ..base_agent import BaseAgent, AgentResult


class DemoAgent(BaseAgent):
    """Agent for constructing examples in tools"""

    AGENT_NAME = "demo_agent"
    AGENT_DESCRIPTION = "a dummy agent for demonstration"
    SYSTEM_PROMPT = ""

    async def construct_core_prompt(self) -> str:
        return ""

    @classmethod
    def generate_examples(cls) -> list[tuple["BaseAgent", AgentResult]]:
        return []
