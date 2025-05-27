# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

from pydantic import Field

from .base_tool import BaseTool
from ..types.tool_types import ToolResult
from ..types.agent_types import AgentInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Calculator(BaseTool):
    TOOL_NAME = "calculate"
    TOOL_DESCRIPTION = """A calculator tool that evaluates mathematical expressions.
Supports basic arithmetic operations (including +, -, *, / and ^) and parentheses.
All expressions must contain only numbers and valid operators."""

    reasoning: str = Field(
        ..., description="Concise resoning about the operation to be performed"
    )
    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate",
        pattern=r"^[\d\s\+\-\*\/\(\)\.]+$",
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        try:
            result = eval(self.expression)
            return ToolResult(
                tool_name=self.TOOL_NAME, success=True, output=str(result)
            )
        except Exception as e:
            return ToolResult(tool_name=self.TOOL_NAME, success=False, errors=str(e))

    @classmethod
    def generate_examples(cls) -> list[tuple["Calculator", ToolResult]]:
        from ..agents.implementations import DemoAgent

        return [
            (
                cls(
                    calling_agent=DemoAgent(),
                    reasoning="The number of fruit is the sum of the two apples and three oranges",
                    expression="2 + 3",
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True, output=str(5)),
            ),
            (
                cls(
                    calling_agent=DemoAgent(),
                    reasoning="The compound expression will require parentheses",
                    expression="(3 * 4) / 2",
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True, output=str(6)),
            ),
        ]


if __name__ == "__main__":
    import asyncio
    from ..agents.implementations import DemoAgent

    async def test():
        c = Calculator(calling_agent=DemoAgent(), reasoning="...", expression="2+2")
        result = await c.run()

        assert result.tool_name == Calculator.TOOL_NAME
        assert result.success
        assert result.duration < 0.5
        assert result.output == str(4)
        print("All tests pass!")

    asyncio.run(test())
