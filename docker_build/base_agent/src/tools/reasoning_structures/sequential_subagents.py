# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A proof of concept / template for a sub-agent based reasoning structure, where
each step is a hard-coded sub-agent call.
"""
import logging

from uuid import uuid4

from ..base_tool import BaseTool
from ...types.tool_types import ToolResult
from ...types.agent_types import AgentInterface
from ...types.llm_types import FCI, ToolCallContent
from ...agents.implementations.coder import CodingAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SubagentBasedReasoningStructure(BaseTool):

    TOOL_NAME = "example_subagent_reasoning_structure"
    TOOL_DESCRIPTION = """Reason through a fixed list of points sequentially."""

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        parent_agent: AgentInterface = self._calling_agent

        try:
            await parent_agent._handle_agent_call(ToolCallContent(
                call_id=f"agent_{uuid4().hex[:8]}",
                tool_name=CodingAgent.AGENT_NAME,
                tool_args=dict(
                    programming_instructions="Print 'a' in a file called 'a.txt'",
                ),
                call_type=FCI.UNCONSTRAINED,  # this must always be UNCONSTRAINED when forcing otherwise it causes 400 errors with the providers.
            ))

            await parent_agent._handle_agent_call(ToolCallContent(
                call_id=f"agent_{uuid4().hex[:8]}",
                tool_name=CodingAgent.AGENT_NAME,
                tool_args=dict(
                    programming_instructions="Print 'b' in a file called 'b.txt'",
                ),
                call_type=FCI.UNCONSTRAINED,  # this must always be UNCONSTRAINED when forcing otherwise it causes 400 errors with the providers.
            ))

            await parent_agent._handle_agent_call(ToolCallContent(
                call_id=f"agent_{uuid4().hex[:8]}",
                tool_name=CodingAgent.AGENT_NAME,
                tool_args=dict(
                    programming_instructions="Print 'c' in a file called 'c.txt'",
                ),
                call_type=FCI.UNCONSTRAINED,  # this must always be UNCONSTRAINED when forcing otherwise it causes 400 errors with the providers.
            ))

            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=True,
                output="Completed successfully"
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=False,
                errors=f"Error in sequential reasoning: {e}"
            )

    @classmethod
    def generate_examples(cls) -> list[tuple["SubagentBasedReasoningStructure", ToolResult]]:
        from ...agents.implementations import DemoAgent

        return [
            (
                cls(calling_agent=DemoAgent()),
                ToolResult(
                    tool_name=cls.TOOL_NAME,
                    success=True,
                    output="Successfully did the ABC",
                ),
            ),
        ]
