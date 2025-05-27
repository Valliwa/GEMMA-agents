# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the Sequential reasoning structure tool."""
import pytest
from unittest.mock import patch, AsyncMock

from src.tools.reasoning_structures.sequential import (
    ToolBasedReasoningStructure, Step, InvocationState, _make_id
)
from src.agents.implementations import DemoAgent
from src.types.tool_types import ToolResult


# Do not use global pytestmark
# Apply asyncio marker only to functions that need it
@pytest.mark.asyncio
async def test_initialization():
    """Test proper initialization of the reasoning structure."""
    structure = ToolBasedReasoningStructure(calling_agent=DemoAgent())
    
    # Verify basic properties
    assert structure.TOOL_NAME == "example_reasoning_structure"
    assert hasattr(structure, "_steps")
    assert len(structure._steps) > 0
    assert all(isinstance(step, Step) for step in structure._steps)


@pytest.mark.asyncio
async def test_run_initializes_state():
    """Test that run() correctly initializes state."""
    agent = DemoAgent()
    structure = ToolBasedReasoningStructure(calling_agent=agent)
    
    # Run the reasoning structure
    result = await structure.run()
    
    # Verify state initialization
    assert len(agent._local_state) == 1
    
    invocation_id = next(iter(agent._local_state.keys()))
    invocation = agent._local_state[invocation_id]
    
    assert isinstance(invocation, InvocationState)
    assert invocation.steps == structure._steps
    assert invocation.current_step_id == structure._steps[0].identifier
    assert invocation.current_step_complete_tool is not None


@pytest.mark.asyncio
async def test_run_registers_completion_tool():
    """Test that run() registers a completion tool for the first step."""
    # Create an empty mock registry
    mock_registry = {}
    
    # Apply the patch within the test
    with patch("src.tools.reasoning_structures.sequential.tool_registry", mock_registry):
        agent = DemoAgent()
        structure = ToolBasedReasoningStructure(calling_agent=agent)
        
        # Run the reasoning structure
        await structure.run()
        
        # Verify a tool was registered
        assert len(mock_registry) == 1
        
        # Get the registered tool
        tool_name = next(iter(mock_registry.keys()))
        
        # Verify it's a completion tool
        assert tool_name.endswith("_complete")
        assert mock_registry[tool_name] in agent._available_tools


@pytest.mark.asyncio
async def test_run_returns_correct_result():
    """Test that run() returns the expected result structure."""
    structure = ToolBasedReasoningStructure(calling_agent=DemoAgent())
    
    # Run the reasoning structure
    result = await structure.run()
    
    # Verify result properties
    assert isinstance(result, ToolResult)
    assert result.tool_name == structure.TOOL_NAME
    assert result.success is True
    assert "step id" in result.output.lower()
    assert "step instructions" in result.output.lower()


@pytest.mark.asyncio
async def test_step_completion_tool_creation():
    """Test the creation of step completion tools."""
    # Setup a mock for create_step_tool
    with patch("src.tools.reasoning_structures.sequential.create_step_tool") as mock_create_step_tool:
        # Setup mock return value
        mock_tool_cls = AsyncMock()
        mock_create_step_tool.return_value = mock_tool_cls
        
        # Create and run structure
        structure = ToolBasedReasoningStructure(calling_agent=DemoAgent())
        await structure.run()
        
        # Verify tool creation
        mock_create_step_tool.assert_called_once()
        
        # Check arguments
        args = mock_create_step_tool.call_args[0]
        assert isinstance(args[0], str)  # invocation_id
        assert isinstance(args[1], Step)  # step

# No asyncio marker for this function since it's synchronous
def test_step_creation_utility():
    """Test the utility function for creating step identifiers."""
    # Generate IDs with custom prefix
    ids = [_make_id("test_prefix") for _ in range(5)]
    
    # Verify uniqueness
    assert len(ids) == len(set(ids))
    
    # Verify format
    for id in ids:
        assert id.startswith("test_prefix_")
        assert len(id) > len("test_prefix_")
    
    # Verify default prefix works
    default_id = _make_id()
    assert default_id.startswith("step_")
