# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the ExecuteCommand tool."""
import pytest
import asyncio
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch

from src.tools.execute_command import ExecuteCommand
from src.types.tool_types import ToolResult
from src.types.agent_types import AgentInterface


class TestExecuteCommand:
    """Test suite for ExecuteCommand tool."""

    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing."""
        mock = Mock(spec=AgentInterface)
        mock._id = "test_agent_id"
        return mock

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Set up a temporary directory for testing."""
        test_dir = tmp_path / "test_dir"
        test_dir.mkdir()
        test_file = test_dir / "test_file.txt"
        test_file.write_text("test content")
        return test_dir

    @pytest.mark.asyncio
    async def test_basic_command_execution(self, mock_agent, temp_dir):
        """Test successful execution of a simple command."""
        # This is an integration test using a real command
        tool = ExecuteCommand(
            calling_agent=mock_agent,
            intent="List files in directory",
            directory_in_which_to_run_command=str(temp_dir),
            command="ls -la",
            command_returns=True,
            generous_expected_duration=5.0,
        )

        result = await tool.run()

        # Verify the result
        assert result.success is True
        assert result.tool_name == ExecuteCommand.TOOL_NAME
        # Verify the command output contains expected content
        assert "test_file.txt" in result.output
        assert "<stdout>" in result.output
        assert "<stderr>" in result.output
        assert "<exit_code>0</exit_code>" in result.output

    @pytest.mark.asyncio
    async def test_working_directory(self, mock_agent, temp_dir):
        """Test that the working directory is set correctly."""
        # Create a nested directory structure
        nested_dir = temp_dir / "nested"
        nested_dir.mkdir()

        # Create a unique file in the nested directory
        unique_file = nested_dir / "unique_file.txt"
        unique_file.write_text("unique content")

        # Run command in the nested directory
        tool = ExecuteCommand(
            calling_agent=mock_agent,
            intent="Check current directory and list files",
            directory_in_which_to_run_command=str(nested_dir),
            command="pwd && ls -l",
            command_returns=True,
            generous_expected_duration=5.0,
        )

        result = await tool.run()

        # Verify results
        assert result.success is True
        assert str(nested_dir) in result.output  # pwd output
        assert "unique_file.txt" in result.output  # ls output

    @pytest.mark.asyncio
    async def test_command_failure(self, mock_agent):
        """Test handling of commands that fail with non-zero exit code."""
        tool = ExecuteCommand(
            calling_agent=mock_agent,
            intent="Run command that fails",
            command="ls /nonexistent_directory",
            command_returns=True,
            generous_expected_duration=5.0,
        )

        result = await tool.run()

        # The tool should succeed in running the command
        assert result.success is True
        # But the command itself should fail
        assert "<exit_code>2</exit_code>" in result.output or "<exit_code>1</exit_code>" in result.output
        assert "No such file or directory" in result.output

    @pytest.mark.asyncio
    async def test_non_returning_command_rejected(self, mock_agent):
        """Test that non-returning commands are rejected."""
        tool = ExecuteCommand(
            calling_agent=mock_agent,
            intent="Start a server",
            command="python -m http.server",
            command_returns=False,  # Explicitly marked as non-returning
            generous_expected_duration=10.0,
        )

        result = await tool.run()

        assert result.success is False
        assert "only supports commands that return" in result.errors

    @pytest.mark.asyncio
    async def test_command_timeout(self, mock_agent, temp_dir):
        """Test that commands timeout properly."""
        # Use the valid temp_dir created by pytest fixture rather than a non-existent workdir
        # Create a command that will run longer than the timeout
        tool = ExecuteCommand(
            calling_agent=mock_agent,
            intent="Sleep command",
            directory_in_which_to_run_command=str(temp_dir),  # Use the temp_dir fixture which we know exists
            command="sleep 10",
            command_returns=True,
            generous_expected_duration=5.1,  # Slightly above minimum to ensure it's valid
        )

        # Create a patched version where we mock the process communication
        with patch.object(asyncio, 'wait_for') as mock_wait_for:
            # Make wait_for raise a timeout error
            mock_wait_for.side_effect = asyncio.TimeoutError()

            # Run the tool with the patched timeout
            result = await tool.run()

            # Now we can more reliably assert the timeout behavior
            assert result.success is False
            assert "timed out" in result.errors.lower()

    @pytest.mark.asyncio
    async def test_prepare_command_with_special_chars(self, mock_agent, temp_dir):
        """Test that command preparation handles special characters properly."""
        # Create a test script that echoes a complex string
        test_script = temp_dir / "test_script.sh"
        test_script.write_text('#!/bin/bash\necho "Special chars: ; > | & $()"\n')
        test_script.chmod(0o755)  # Make executable

        tool = ExecuteCommand(
            calling_agent=mock_agent,
            intent="Test special characters",
            directory_in_which_to_run_command=str(temp_dir),
            command="./test_script.sh",
            command_returns=True,
            generous_expected_duration=5.0,
        )

        result = await tool.run()

        assert result.success is True
        # Check for partial special characters - the actual output may vary based on shell escaping
        assert "Special chars:" in result.output
        assert all(char in result.output for char in [";", ">", "|", "&"])
        assert "<exit_code>0</exit_code>" in result.output

    @pytest.mark.asyncio
    async def test_multiple_commands(self, mock_agent, temp_dir):
        """Test executing multiple commands in a sequence."""
        tool = ExecuteCommand(
            calling_agent=mock_agent,
            intent="Run multiple commands",
            directory_in_which_to_run_command=str(temp_dir),
            command="echo 'First command' && echo 'Second command'",
            command_returns=True,
            generous_expected_duration=5.0,
        )

        result = await tool.run()

        assert result.success is True
        assert "First command" in result.output
        assert "Second command" in result.output
        assert "<exit_code>0</exit_code>" in result.output

    @pytest.mark.asyncio
    async def test_environment_variables(self, mock_agent, temp_dir):
        """Test that environment variables are properly accessible."""
        # Create a test environment variable
        env_var_name = "TEST_ENV_VAR_FOR_EXEC_COMMAND"
        env_var_value = "test_value_12345"
        os.environ[env_var_name] = env_var_value

        tool = ExecuteCommand(
            calling_agent=mock_agent,
            intent="Test environment variable access",
            directory_in_which_to_run_command=str(temp_dir),  # Use the temp_dir fixture which we know exists
            command=f"echo ${env_var_name}",
            command_returns=True,
            generous_expected_duration=5.0,
        )

        result = await tool.run()

        # Clean up the environment variable
        del os.environ[env_var_name]

        assert result.success is True
        assert env_var_value in result.output

    @pytest.mark.asyncio
    async def test_command_with_large_output(self, mock_agent, temp_dir):
        """Test handling commands that produce large outputs."""
        # Generate a command that produces a large output
        command = "for i in {1..1000}; do echo \"Line $i of output\"; done"

        tool = ExecuteCommand(
            calling_agent=mock_agent,
            intent="Test large output handling",
            directory_in_which_to_run_command=str(temp_dir),  # Use the temp_dir fixture which we know exists
            command=command,
            command_returns=True,
            generous_expected_duration=10.0,
        )

        result = await tool.run()

        assert result.success is True
        assert "Line 1 of output" in result.output
        assert "Line 1000 of output" in result.output

    # This test uses mocking because it's testing a specific interaction with the process
    @pytest.mark.asyncio
    async def test_process_termination_on_timeout(self, mock_agent, temp_dir):
        """Test that processes are terminated properly on timeout."""
        async def mock_communicate():
            await asyncio.sleep(10)  # This will be cancelled by the timeout
            return b"", b""

        async def mock_wait():
            return 0

        # Create a patched version where we mock the process communication
        mock_process = Mock()
        mock_process.terminate = Mock()
        mock_process.wait = mock_wait
        mock_process.kill = Mock()
        mock_process.communicate = mock_communicate  # Directly use the coroutine

        with patch('asyncio.create_subprocess_shell', new=Mock()) as mock_create:
            # Set up process creation to return our mocked process
            create_future = asyncio.Future()
            create_future.set_result(mock_process)
            mock_create.return_value = create_future

            # Mock wait_for to simulate a timeout
            with patch('asyncio.wait_for') as mock_wait_for:
                mock_wait_for.side_effect = asyncio.TimeoutError()

                tool = ExecuteCommand(
                    calling_agent=mock_agent,
                    intent="Test termination",
                    directory_in_which_to_run_command=str(temp_dir),  # Use the temp_dir fixture
                    command="sleep 100",
                    command_returns=True,
                    generous_expected_duration=5.0,  # Use minimum valid duration
                )

                result = await tool.run()

                # Verify the result
                assert result.success is False
                assert "timed out" in result.errors.lower()
                # Verify termination was called
                mock_process.terminate.assert_called_once()
