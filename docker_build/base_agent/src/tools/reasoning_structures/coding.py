# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A reasoning structure for coding.
"""
import logging

from pydantic import PrivateAttr

from .sequential import Step, ToolBasedReasoningStructure, _make_id
from ...types.tool_types import ToolResult
from ...types.agent_types import AgentInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class CodingReasoningStructure(ToolBasedReasoningStructure):

    TOOL_NAME = "coding_reasoning_structure"
    TOOL_DESCRIPTION = """Apply this reasoning structure when you detect that you have a non-trivial coding implementation task that requires a methodical approach involving initial exploration, implementation, verification and cleanup to complete well.

Do not call this tool if merely verifying, testing or if the task at hand is quick and does not require such rigour.

This reasoning structure will guide you through good software engineering practices, and ensure that no steps have been missed out.
"""
    _steps: list[Step] = PrivateAttr(default_factory=lambda: [
        Step(
            identifier=_make_id(),
            instruction="Explore the project to a) locate all useful documentation (README.md files, common likely MD documentation files, etc), b) all files that may relate to your programming instructions, c) identify module-level and file-level design patterns and conventions.",
            done_description="You have viewed each of these files, made sure to close irrelevant or long files, and taken notes or summaries. Note that for greenfiled projects, this step may complete trivially.",
            failed_description="Files could not be opened for some reason, or the project location is unclear.",
        ),
        Step(
            identifier=_make_id(),
            instruction="Carefully implement the solution completely and thoroughly. Make sure you observe any existing stylistic conventions, and effectively re-use existing design patterns or modules to avoid duplicating functionality.",
            done_description="A first pass at the code implementation has been implemented, with tests not yet having been run.",
            failed_description="You have got stuck trying to get dependencies set up, getting mocks and fixtures set up, or have otherwise digressed from the core code implementation.",
        ),
        Step(
            identifier=_make_id(),
            instruction="Test the implementation end-to-end, favouring test scripts instead of test frameworks. If this is not an option or the project already has a test framework set up, then use that.",
            done_description="You have ensured that the code is valid, hasn't introduced any regressions and works as intended",
            failed_description="You have got stuck writing TDD loops, getting dependencies set up, getting mocks and fixtures set up",
        ),
        Step(
            identifier=_make_id(),
            instruction="Clean up: remove any temporary test scripts, toy implementations or other scaffolding. Check that all documentation and docstrings are up-to-date.",
            done_description="All temporary files have been removed, and documentation updated.",
        ),
    ])

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    @classmethod
    def generate_examples(cls) -> list[tuple["CodingReasoningStructure", ToolResult]]:
        from ...agents.implementations import DemoAgent

        return [
            (
                cls(calling_agent=DemoAgent()),
                ToolResult(
                    tool_name=cls.TOOL_NAME,
                    success=True,
                    output="The first step in the meta improvement process is: ...",
                ),
            ),
        ]
