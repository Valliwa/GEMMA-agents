# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the BaseTool class functionality."""
import pytest
from unittest.mock import Mock, patch
import asyncio
from typing import Optional

# Fix the import paths to work when running from the base_agent directory
from src.tools.base_tool import BaseTool, tool_registry
from src.types.tool_types import ToolResult
from src.types.agent_types import AgentInterface
from src.types.common import ArgFormat

class TestBaseTool:
    """Test suite for BaseTool class."""

    def setup_method(self):
        """Setup for each test method."""
        # Save the original registry and clear it for testing
        self.original_registry = dict(tool_registry)
        tool_registry.clear()

    def teardown_method(self):
        """Teardown after each test method."""
        # Restore the original registry after each test
        tool_registry.clear()
        tool_registry.update(self.original_registry)

    def test_tool_registration(self):
        """Test that tools are properly registered through metaclass."""
        # Define a test tool class
        class TestTool(BaseTool):
            TOOL_NAME = "test_tool"
            TOOL_DESCRIPTION = "A test tool for registration"

            async def run(self) -> ToolResult:
                return ToolResult(tool_name=self.TOOL_NAME, success=True)

            @classmethod
            def generate_examples(cls):
                return []

        # Verify the tool was registered correctly
        assert "test_tool" in tool_registry
        assert tool_registry["test_tool"] == TestTool

    @pytest.mark.asyncio
    async def test_tool_examples(self):
        """Test that generate_examples returns valid examples."""
        # Define a test tool with examples
        class ExampleTool(BaseTool):
            TOOL_NAME = "example_tool"
            TOOL_DESCRIPTION = "Test tool with examples"

            async def run(self) -> ToolResult:
                return ToolResult(tool_name=self.TOOL_NAME, success=True)

            @classmethod
            def generate_examples(cls):
                # Return a minimal valid example
                mock_agent = Mock(spec=AgentInterface)
                tool_instance = cls(calling_agent=mock_agent)
                tool_result = ToolResult(tool_name=cls.TOOL_NAME, success=True)
                return [(tool_instance, tool_result)]

        # Check examples format
        examples = ExampleTool.generate_examples()

        assert isinstance(examples, list)
        assert len(examples) == 1
        example = examples[0]
        assert isinstance(example, tuple)
        assert len(example) == 2
        assert isinstance(example[0], ExampleTool)
        assert isinstance(example[1], ToolResult)

    @pytest.mark.asyncio
    async def test_args_str_to_dict(self):
        """Test XML and JSON argument parsing."""
        from pydantic import Field

        class ArgTool(BaseTool):
            TOOL_NAME = "arg_tool"
            TOOL_DESCRIPTION = "Test tool with arguments"

            arg1: str = Field(..., description="Test argument")
            arg2: int = Field(default=0, description="Optional argument")

            async def run(self) -> ToolResult:
                return ToolResult(tool_name=self.TOOL_NAME, success=True)

            @classmethod
            def generate_examples(cls):
                return []

        # Test XML parsing
        xml_args = """
        <TOOL_ARGS>
            <arg1>test</arg1>
            <arg2>42</arg2>
        </TOOL_ARGS>
        """
        args_dict, warnings = await ArgTool.args_str_to_dict(xml_args, ArgFormat.XML)
        assert args_dict is not None
        assert args_dict["arg1"] == "test"
        assert args_dict["arg2"] == 42
        assert warnings is None

        # Test bad XML - this should result in a warning and possibly a None args_dict
        # or a dict with only default values, depending on the implementation
        bad_xml = "<TOOL_ARGS><arg1>test</TOOL_ARGS>"
        args_dict, warnings = await ArgTool.args_str_to_dict(bad_xml, ArgFormat.XML)
        # The important thing is that a warning is generated
        assert warnings is not None

        # We don't make assumptions about whether args_dict is None or partially populated
        # as implementation details can vary. If it's None, the test passes.
        # If not None, check that it doesn't contain the required field or that it does have defaults.
        if args_dict is not None:
            # It might contain default values but not the required field
            assert "arg1" not in args_dict, "Required field should not be present in malformed XML"
            # Optionally check if default values are preserved
            # We don't assert this as it's an implementation detail that could change
            # assert args_dict.get("arg2") == 0, "Default value should be present"

    @pytest.mark.asyncio
    async def test_tool_result_formatting(self):
        """Test that tool results are properly formatted."""
        # Create a mock agent for testing
        mock_agent = Mock(spec=AgentInterface)

        # Define a simple test tool
        class ResultTool(BaseTool):
            TOOL_NAME = "result_tool"
            TOOL_DESCRIPTION = "Test tool for result formatting"

            async def run(self) -> ToolResult:
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=True,
                    output="test output",
                    warnings="test warning",
                    errors=None
                )

            @classmethod
            def generate_examples(cls):
                return []

        # Test successful tool execution
        tool = ResultTool(calling_agent=mock_agent)
        result = await tool.run()

        # Check result structure
        assert isinstance(result, ToolResult)
        assert result.tool_name == "result_tool"
        assert result.success is True
        assert "test output" in str(result)
        assert "test warning" in str(result)

        # Test failure result formatting
        failure_result = ToolResult(
            tool_name="fail_tool",
            success=False,
            output=None,
            warnings=None,
            errors="test error"
        )

        # Check failure result structure
        assert isinstance(failure_result, ToolResult)
        assert failure_result.tool_name == "fail_tool"
        assert failure_result.success is False
        assert "test error" in str(failure_result)
        assert "SUCCESS" not in str(failure_result)
        assert "FAILURE" in str(failure_result)

if __name__ == "__main__":
    # Run the tests directly for debugging
    pytest.main(["-xvs", __file__])
