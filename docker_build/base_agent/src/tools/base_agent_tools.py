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


class ReturnResult(BaseTool):
    """Tool for returning a result from a benchmark question."""

    TOOL_NAME = "return_result"
    TOOL_DESCRIPTION = """Store the result to this agent run.

Once you get to the end of your problem solving or execution, use this tool to
register a result on the 'return stack'. You may call this function multiple
times to revise an already registered answer if you change your mind.

The result should be clear and concise and will be provided to the caller of
this agent's execution. If it does not make sense to return an explicit value,
then provide a concise summary of what you did during the run. Otherwise,
submit the complete response that directly addresses the question.

You should avoid including additional content that may confuse the caller.

You MUST call this tool at least once before completing.
"""

    # reasoning: str = Field(
    #     ...,
    #     description="Concise reasoning about the result you are going to submit and the correct format in which to do so",
    # )
    result: str = Field(
        ...,
        description="The complete result you wish to return",
        min_length=1,
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        """Execute the answer submission with parsing."""
        try:
            # Validate answer is not empty or just whitespace
            result = self.result.strip()
            if not result:
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=False,
                    errors="Return value cannot be empty",
                )

            self._calling_agent._local_state['return_value'] = self.result

            return ToolResult(tool_name=self.TOOL_NAME, success=True)

        except Exception as e:
            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=False,
                errors=f"Failed to record result: {str(e)}",
            )

    @classmethod
    async def args_str_to_dict(
        cls, args_str: str, arg_format: ArgFormat = ArgFormat.XML
    ) -> tuple[dict | None, str | None]:
        # Handles the edge case when the result is some numerical value (e.g.
        # 42) which is cast to an integer and fails pydantic field validation
        # which is expecting a string
        args_dict, parse_warnings = await args_str_to_dict(
            args_str, guide_obj=cls, arg_format=arg_format, root_tag="TOOL_ARGS"
        )
        if args_dict:
            args_dict["result"] = str(args_dict["result"])
        return args_dict, parse_warnings

    @classmethod
    def generate_examples(cls) -> list[tuple["ReturnResult", ToolResult]]:
        """Generate example uses of the submit_answer tool."""
        from ..agents.implementations import DemoAgent

        return [
            (
                cls(
                    calling_agent=DemoAgent(),
                    # reasoning="After thoroughly inspecting the file tree, I should return the full filepaths of the two files that we need to fix the memory leak issue",
                    result="""/home/agent/workdir/src/initialise.c
/home/agent/workdir/src/collections/allocator.c""",
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
            (
                cls(
                    calling_agent=DemoAgent(),
                    # reasoning="The speed of the car is 10mph",
                    result="10 miles per hour",
                ),
                ToolResult(
                    tool_name=cls.TOOL_NAME, success=False, errors="Parser error"
                ),
            ),
            (
                cls(
                    calling_agent=DemoAgent(),
                    # reasoning="The calculated value is 1,234.5",
                    result="1,234.5",
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
        ]


class ExitAgent(BaseTool):
    """Tool for early return from an agent"""

    TOOL_NAME = "early_exit"
    TOOL_DESCRIPTION = """Use this tool to exit early, if progress becomes impossible or illogical.

If there seems to be an error, or no logical way to proceed with the execution, then in exceptional circumstances you can call this tool to exit early and return to your caller, if any.

You should only use this tool sparingly, and only if you are sure that this is the most sensible thing to do after having given your task a best-effort attempt.
"""

    exit_reason: str = Field(
        ...,
        description="Concise reasoning as to why exiting now is the best option, which will be passed up to your caller.",
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        """Execute the answer submission with parsing."""
        try:
            # Validate answer is not empty or just whitespace
            reason = self.exit_reason.strip()
            if not reason:
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=False,
                    errors="You must provide a reason for why you're exiting",
                )

            self._calling_agent._local_state["needs_exit"] = True
            self._calling_agent._local_state["exit_reason"] = reason

            return ToolResult(
                tool_name=self.TOOL_NAME, success=True, output=dict(exit_reason=reason)
            )

        except Exception as e:
            logger.info(f"Failed to exit agent: {e}")
            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=False,
                errors=f"Failed to invoke exit. Try returning a result instead.",
            )

    @classmethod
    def generate_examples(cls) -> list[tuple["ExitAgent", ToolResult]]:
        """Generate example uses of the submit_answer tool."""
        from ..agents.implementations import DemoAgent

        return [
            # Example 1: Insufficient permissions
            (
                cls(
                    calling_agent=DemoAgent(),
                    exit_reason="Unable to access the required database. The current credentials lack the necessary permissions to perform this operation.",
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
            # Example 2: Critical prerequisite missing
            (
                cls(
                    calling_agent=DemoAgent(),
                    exit_reason="Required ML model weights file is missing from the specified path. Cannot perform inference without the model.",
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
            # Example 3: External service dependency failure
            (
                cls(
                    calling_agent=DemoAgent(),
                    exit_reason="Required external API service is unavailable after multiple retry attempts. Cannot complete the operation.",
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
            # Example of invalid usage (empty reason)
            (
                cls(
                    calling_agent=DemoAgent(),
                    exit_reason="   ",
                ),
                ToolResult(
                    tool_name=cls.TOOL_NAME,
                    success=False,
                    errors="You must provide a reason for why you're exiting",
                ),
            ),
        ]


class Complete(BaseTool):
    """Tool for early return from an agent"""

    TOOL_NAME = "complete"
    TOOL_DESCRIPTION = """You MUST use this tool to finish execution, once all the tasks have been done, the answer submitted or result returned."""

    all_complete: bool = Field(
        ...,
        description="Whether everything is done: all tasks are completed, results are returned, answers submitted, etc.",
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        """Execute the answer submission with parsing."""
        try:
            if self.all_complete:
                self._calling_agent._local_state["exec_complete"] = True
                return ToolResult(
                    tool_name=self.TOOL_NAME, success=True,
                )
            else:
                return ToolResult(
                    tool_name=self.TOOL_NAME, success=False, errors="Make sure to complete all tasks before calling this tool again to exit"
                )

        except Exception as e:
            logger.info(f"Failed to exit agent: {e}")
            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=False,
                errors=f"Failed to invoke complete",
            )

    @classmethod
    def generate_examples(cls) -> list[tuple["Complete", ToolResult]]:
        """Generate example uses of the submit_answer tool."""
        from ..agents.implementations import DemoAgent

        return [
            # Example 1: Insufficient permissions
            (
                cls(
                    calling_agent=DemoAgent(),
                    all_complete=True,
                ),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
        ]


class RegenerateContext(BaseTool):
    """Tool for early return from an agent"""

    TOOL_NAME = "regenerate_context"
    TOOL_DESCRIPTION = """Use this tool to consolidate a set of file changes into your FILE_VIEWER.

If you have made a sequence of edits to files, or if files have changed on disk by some other process or agent, and you wish to get a consolidated and up-to-date view of the directory tree and open file contents, then call this tool.

This will update the open files with the current file content, update the directory tree, and remove all in-line file OPEN or EDIT blocks.

Warning: This will break your KV cache, leading to higher costs and latency following this tool call. Only call this tool if it is getting tricky to follow what the state of the files are, if the accumulated edit and file open blocks in your context are getting quite long, or if you suspect the file has changed on disk since last viewing it.
"""
    reasoning: str = Field(..., description="Concise reasoning about why we need to re-generate the context")

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    async def run(self) -> ToolResult:
        """Execute the answer submission with parsing."""
        try:
            self._calling_agent._local_state["should_regenerate"] = True

            return ToolResult(
                tool_name=self.TOOL_NAME, success=True,
            )

        except Exception as e:
            logger.info(f"Failed to trigger regeneration: {e}")
            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=False,
                errors=f"Failed to trigger regeneration: {e}.",
            )

    @classmethod
    def generate_examples(cls) -> list[tuple["RegenerateContext", ToolResult]]:
        """Generate example uses of the submit_answer tool."""
        from ..agents.implementations import DemoAgent

        return [
            (
                cls(calling_agent=DemoAgent(), reasoning="There have been 20 sequential edits on the same file and the context is getting long."),
                ToolResult(tool_name=cls.TOOL_NAME, success=True),
            ),
        ]
