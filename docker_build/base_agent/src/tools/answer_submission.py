# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging

from pydantic import Field

from .base_tool import BaseTool
from ..schemas import args_str_to_dict
from ..types.tool_types import ToolResult
from ..types.agent_types import AgentInterface
from ..types.common import ArgFormat

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SubmitAnswer(BaseTool):
    """Tool for submitting answers to benchmark questions on disk.

    This is slightly different to the ReturnResult tool which is used to return
    a result from the end of agent function call.
    """

    TOOL_NAME = "submit_answer"
    TOOL_DESCRIPTION = """Submit an answer to a benchmark question. The answer should be clear and concise.
The tool will attempt to parse your answer according to the benchmark's requirements.
Your answer should be a complete response that directly addresses the question.
It is very important that you do not include any extraneous words or content in the answer field that may make the parsing fail.
"""

    # reasoning: str = Field(
    #     ...,
    #     description="Reason about the answer you are going to submit and the correct format in which to do so",
    # )

    answer: str = Field(
        ..., description="Your complete answer to the benchmark question", min_length=1
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        """Execute the answer submission with parsing."""
        try:
            if not self._calling_agent._logdir:
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=False,
                    errors="System error: no answer path available",
                )

            # Validate answer is not empty or just whitespace
            answer = self.answer.strip()
            if not answer:
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=False,
                    errors="Answer cannot be empty",
                )

            # Save answer to disk
            path = self._calling_agent._logdir / "answer.txt"
            with open(path, "w") as f:
                f.write(answer)

            return ToolResult(tool_name=self.TOOL_NAME, success=True)

        except Exception as e:
            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=False,
                errors=f"Failed to save answer: {str(e)}",
            )

    @classmethod
    async def args_str_to_dict(
        cls, args_str: str, arg_format: ArgFormat = ArgFormat.XML
    ) -> tuple[dict | None, str | None]:
        args_dict, parse_warnings = await args_str_to_dict(
            args_str, guide_obj=cls, arg_format=arg_format, root_tag="TOOL_ARGS"
        )
        if args_dict:
            args_dict["answer"] = str(args_dict["answer"])
        return args_dict, parse_warnings

    @classmethod
    def generate_examples(cls) -> list[tuple["SubmitAnswer", ToolResult]]:
        """Generate example uses of the submit_answer tool."""
        from ..agents.implementations import DemoAgent

        return [
            (
                cls(
                    calling_agent=DemoAgent(),
                    answer="5",
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
            (
                cls(
                    calling_agent=DemoAgent(),
                    # reasoning="The speed of the car is 10mph",
                    answer="10 miles per hour",
                ),
                ToolResult(
                    tool_name=cls.TOOL_NAME, success=False, errors="Parser error"
                ),
            ),
            (
                cls(
                    calling_agent=DemoAgent(),
                    # reasoning="The calculated value is 1,234.5",
                    answer="1,234.5",
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
        ]
