# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the REFUTE benchmark implementation."""
import pytest
import tempfile
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

from src.benchmarks.refute import RefuteBenchmark, RefuteExample


class TestRefuteExample:
    """Tests for the RefuteExample class."""

    def test_from_raw(self):
        """Test conversion from raw dataset example."""
        # Create a mock raw example
        raw_example = {
            "problem_id": "test_problem",
            "statement": "Find a counterexample for the given algorithm",
            "problem_rating": 1500,
            "author_rating": 1800,
            "input": "Integer n followed by n integers",
            "output": "Yes or No",
            "tags": "['math', 'algorithms']",
            "wrong_code": "def solution(n, arr):\n    return 'Yes'",
            "wrong_code_lang": "Python code",
            "correct_cpp": "// C++ solution",
            "correct_python": "def solution(n, arr):\n    # some checks\n    return 'Yes' if condition else 'No'",
            "validator": "def validate(input):\n    return True",
            "example_input": "3\n1 2 3",
            "example_output": "Yes",
            "note": "Find a case where answer should be 'No'",
            "time_limit": 3000,  # In milliseconds
            "memory_limit": 256,
            "fail_tc": 5
        }

        example = RefuteExample.from_raw(raw_example)

        # Verify basic properties
        assert example.problem_id == "test_problem"
        assert example.problem_statement == "Find a counterexample for the given algorithm"
        assert example.problem_rating == 1500
        assert example.author_rating == 1800
        assert example.input_format == "Integer n followed by n integers"
        assert example.output_format == "Yes or No"
        assert example.tags == ["math", "algorithms"]
        assert example.incorrect_code == "def solution(n, arr):\n    return 'Yes'"
        assert example.incorrect_code_lang == "Python code"
        assert example.correct_cpp == "// C++ solution"
        assert example.correct_python == "def solution(n, arr):\n    # some checks\n    return 'Yes' if condition else 'No'"
        assert example.validator == "def validate(input):\n    return True"
        assert example.example_input == "3\n1 2 3"
        assert example.example_output == "Yes"
        assert example.note == "Find a case where answer should be 'No'"
        assert example.time_limit == 3.0  # Converted to seconds
        assert example.memory_limit == 256
        assert example.fail_tc == 5

    def test_format_problem_statement(self):
        """Test formatting of the complete problem statement."""
        test_example = RefuteExample(
            problem_id="test_problem",
            problem_statement="Test a sorting algorithm",
            problem_rating=1500,
            author_rating=1800,
            input_format="Integer n followed by n integers",
            output_format="Sorted list of integers",
            tags=["algorithms", "sorting"],
            incorrect_code="def sort(arr):\n    return arr  # doesn't sort",
            incorrect_code_lang="Python",
            correct_cpp="// corrected C++ code",
            correct_python="def sort(arr):\n    return sorted(arr)",
            validator="def validate(input_str):\n    return True",
            example_input="3\n3 1 2",
            example_output="1 2 3",
            note="Find inputs where the incorrect sort fails",
            time_limit=2.0,
            memory_limit=128,
            fail_tc=None
        )

        formatted = test_example.format_problem_statement()

        # Verify the formatting contains key elements
        assert "# REFUTE Challenge: Create a Counterexample" in formatted
        assert "## Problem ID: test_problem" in formatted
        assert "## Problem Rating: 1500" in formatted
        assert "## Tags: algorithms, sorting" in formatted
        assert "## Problem Statement:" in formatted
        assert "Test a sorting algorithm" in formatted
        assert "## Input Format:" in formatted
        assert "Integer n followed by n integers" in formatted
        assert "## Output Format:" in formatted
        assert "Sorted list of integers" in formatted
        assert "## Example Input:" in formatted
        assert "3\n3 1 2" in formatted
        assert "## Example Output:" in formatted
        assert "1 2 3" in formatted
        assert "## Incorrect Solution (Python):" in formatted
        assert "```python" in formatted
        assert "def sort(arr):" in formatted
        assert "## Your Task:" in formatted
        assert "Create a counterexample" in formatted


@patch("src.benchmarks.refute.load_dataset")
def test_benchmark_initialization(mock_load_dataset):
    """Test initialization of the RefuteBenchmark."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = [
        {
            "problem_id": f"problem_{i}",
            "statement": f"Test problem {i}",
            "problem_rating": 1500 + i * 100,
            "author_rating": 1800,
            "input": "Input format",
            "output": "Output format",
            "tags": "['math']",
            "wrong_code": "def solution():\n    pass",
            "wrong_code_lang": "Python",
            "correct_cpp": "// C++ solution",
            "correct_python": "def solution():\n    return 'correct'",
            "validator": "def validate(input):\n    return True",
            "example_input": "Example input",
            "example_output": "Example output",
            "time_limit": 3000,
            "memory_limit": 256
        }
        for i in range(1, 6)  # 5 example problems
    ]
    mock_load_dataset.return_value = mock_dataset

    # Initialize benchmark
    benchmark = RefuteBenchmark(seed=42, subset_size=3)

    # Verify benchmark properties
    assert benchmark.name == "refute"
    assert len(benchmark.problems) == 3  # Should respect subset_size

    # Verify problems have the expected structure
    for problem in benchmark.problems:
        assert isinstance(problem.statement, str)
        assert "REFUTE Challenge" in problem.statement
        assert isinstance(problem.answer, RefuteExample)
        assert problem.answer_discussion is None


@pytest.mark.asyncio
@patch("src.benchmarks.refute.load_dataset")
@patch("src.benchmarks.refute.subprocess.run")
async def test_setup_problem(mock_subprocess_run, mock_load_dataset):
    """Test the setup_problem method."""
    # Create mock dataset with simpler structure
    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = [
        {
            "problem_id": "setup_test",
            "statement": "Test problem",
            "problem_rating": 1500,
            "author_rating": 1800,
            "input": "Input format",
            "output": "Output format",
            "tags": "['math']",
            "wrong_code": "def solution() {\n  return false;\n}",  # C++ code
            "wrong_code_lang": "C++",  # Set to trigger C++ compilation
            "correct_cpp": "// C++ solution",
            "correct_python": "def solution():\n    return True",
            "validator": "def validate(input):\n    return True",
            "example_input": "Example input",
            "example_output": "Example output"
        }
    ]
    mock_load_dataset.return_value = mock_dataset

    # Mock subprocess run to handle compilation
    def mock_run_impl(*args, **kwargs):
        # Create a mock result object
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result

    mock_subprocess_run.side_effect = mock_run_impl

    # Initialize benchmark and get first problem
    benchmark = RefuteBenchmark(seed=42, subset_size=1)
    problem = benchmark.problems[0]

    # Create temporary directory for problem data
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Run setup_problem method
        await benchmark.setup_problem(
            problem=problem,
            problem_data_dir=tmp_path,
            container_name="test_container"
        )

        # Verify directory structure and files
        validator_dir = tmp_path / "validator"
        assert validator_dir.exists()

        # Check validator script
        validator_path = validator_dir / "validator.py"
        assert validator_path.exists()
        assert validator_path.read_text() == "def validate(input):\n    return True"

        # Check incorrect solution setup (C++ files)
        incorrect_solution_cpp = validator_dir / "incorrect_solution.cpp"
        incorrect_solution_bin = validator_dir / "incorrect_solution"
        assert incorrect_solution_cpp.exists()
        assert "return false;" in incorrect_solution_cpp.read_text().lower()

        # Check correct solution (Python file)
        correct_solution_path = validator_dir / "correct_solution.py"
        assert correct_solution_path.exists()
        assert correct_solution_path.read_text() == "def solution():\n    return True"

        # Check verification script
        verify_path = tmp_path / "verify_counterexample.py"
        assert verify_path.exists()
        assert os.access(verify_path, os.X_OK)  # Should be executable

        # Verify README
        readme_path = tmp_path / "README.md"
        assert readme_path.exists()
        assert "REFUTE Challenge" in readme_path.read_text()

        # Verify mock compilation was called appropriately
        # C++ compilation command should be called
        expected_compile_cmd = [
            "docker", "exec", "test_container",
            "g++", "-std=c++17", "-O2",
            f"/home/agent/workdir/validator/incorrect_solution.cpp",
            "-o", "/home/agent/workdir/validator/incorrect_solution"
        ]

        mock_subprocess_run.assert_called_with(
            expected_compile_cmd,
            check=True,
            capture_output=True
        )


@patch("src.benchmarks.refute.load_dataset")
@patch("src.benchmarks.refute.subprocess.run")
@pytest.mark.asyncio
async def test_score_problem_valid_counterexample(mock_subprocess_run, mock_load_dataset):
    """Test scoring a valid counterexample."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = [
        {
            "problem_id": "score_test",
            "statement": "Test problem",
            "problem_rating": 1500,
            "author_rating": 1800,
            "input": "Input format",
            "output": "Output format",
            "tags": "['math']",
            "wrong_code": "def solution():\n    pass",
            "wrong_code_lang": "Python",
            "correct_cpp": "// C++ solution",
            "correct_python": "def solution():\n    return 'correct'",
            "validator": "def validate(input):\n    return True",
            "example_input": "Example input",
            "example_output": "Example output"
        }
    ]
    mock_load_dataset.return_value = mock_dataset

    # Setup subprocess.run to return different outputs for incorrect and correct solutions
    def mock_run_impl(args, **kwargs):
        result = MagicMock()
        result.returncode = 0

        # If executing validator, return success
        if "validator.py" in " ".join(args):
            result.returncode = 0
            return result

        # If executing incorrect solution, return one output
        if "incorrect_solution.py" in " ".join(args):
            result.stdout = "Incorrect output"
            return result

        # If executing correct solution, return different output
        if "correct_solution.py" in " ".join(args):
            result.stdout = "Correct output"
            return result

        return result

    mock_subprocess_run.side_effect = mock_run_impl

    # Initialize benchmark
    benchmark = RefuteBenchmark(seed=42, subset_size=1)
    problem = benchmark.problems[0]

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        workdir_path = tmp_path / "workdir"
        workdir_path.mkdir()
        validator_dir = workdir_path / "validator"
        validator_dir.mkdir()

        answer_dir_path = tmp_path / "answer_dir"
        answer_dir_path.mkdir()

        # Create answer.txt with counterexample
        answer_txt_path = answer_dir_path / "answer.txt"
        answer_txt_path.write_text("3\n1 2 3")

        # Score the problem
        score, errors, discussion = await benchmark.score_problem(
            problem=problem,
            agent_workdir=str(workdir_path),
            agent_answer_dir=str(answer_dir_path),
            container_name="test_container"
        )

        # Verify the scoring
        assert score == 1.0
        assert errors is None
        assert "Valid counterexample found" in discussion


@patch("src.benchmarks.refute.load_dataset")
@patch("src.benchmarks.refute.subprocess.run")
@pytest.mark.asyncio
async def test_score_problem_invalid_counterexample(mock_subprocess_run, mock_load_dataset):
    """Test scoring an invalid counterexample (same outputs)."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = [
        {
            "problem_id": "score_test",
            "statement": "Test problem",
            "problem_rating": 1500,
            "author_rating": 1800,
            "input": "Input format",
            "output": "Output format",
            "tags": "['math']",
            "wrong_code": "def solution():\n    pass",
            "wrong_code_lang": "Python",
            "correct_cpp": "// C++ solution",
            "correct_python": "def solution():\n    return 'correct'",
            "validator": "def validate(input):\n    return True",
            "example_input": "Example input",
            "example_output": "Example output"
        }
    ]
    mock_load_dataset.return_value = mock_dataset

    # Setup subprocess.run to return same output for both solutions
    def mock_run_impl(args, **kwargs):
        result = MagicMock()
        result.returncode = 0

        # Return successful validation
        if "validator.py" in " ".join(args):
            result.returncode = 0
            return result

        # Return same output for both solutions (not a valid counterexample)
        result.stdout = "Same output"
        return result

    mock_subprocess_run.side_effect = mock_run_impl

    # Initialize benchmark
    benchmark = RefuteBenchmark(seed=42, subset_size=1)
    problem = benchmark.problems[0]

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        workdir_path = tmp_path / "workdir"
        workdir_path.mkdir()
        validator_dir = workdir_path / "validator"
        validator_dir.mkdir()

        answer_dir_path = tmp_path / "answer_dir"
        answer_dir_path.mkdir()

        # Create answer.txt with counterexample
        answer_txt_path = answer_dir_path / "answer.txt"
        answer_txt_path.write_text("3\n1 2 3")

        # Score the problem
        score, errors, discussion = await benchmark.score_problem(
            problem=problem,
            agent_workdir=str(workdir_path),
            agent_answer_dir=str(answer_dir_path),
            container_name="test_container"
        )

        # Verify the scoring
        assert score == 0.0
        assert errors == "Not a valid counterexample"
        assert "Both solutions produced the same output" in discussion

@patch("src.benchmarks.refute.load_dataset")
@patch("src.benchmarks.refute.subprocess.run")
@pytest.mark.asyncio
async def test_score_problem_validation_failure(mock_subprocess_run, mock_load_dataset):
    """Test scoring when input validation fails."""
    # Create mock dataset
    mock_dataset = MagicMock()
    mock_dataset.__getitem__.return_value = [
        {
            "problem_id": "validation_test",
            "statement": "Test problem",
            "problem_rating": 1500,
            "author_rating": 1800,
            "input": "Input format",
            "output": "Output format",
            "tags": "['math']",
            "wrong_code": "def solution():\n    pass",
            "wrong_code_lang": "Python",
            "correct_cpp": "// C++ solution",
            "correct_python": "def solution():\n    return 'correct'",
            "validator": "def validate(input):\n    return False",  # Validation always fails
            "example_input": "Example input",
            "example_output": "Example output"
        }
    ]
    mock_load_dataset.return_value = mock_dataset

    # Setup subprocess.run to simulate validation failure
    def mock_run_impl(args, **kwargs):
        result = MagicMock()

        # Validation failure
        if "validator.py" in " ".join(args):
            result.returncode = 1
            result.stderr = "Validation error: Invalid input format"
            return result

        result.returncode = 0
        return result

    mock_subprocess_run.side_effect = mock_run_impl

    # Initialize benchmark
    benchmark = RefuteBenchmark(seed=42, subset_size=1)
    problem = benchmark.problems[0]

    # Create temporary directory structure
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        workdir_path = tmp_path / "workdir"
        workdir_path.mkdir()

        answer_dir_path = tmp_path / "answer_dir"
        answer_dir_path.mkdir()

        # Create answer.txt with counterexample
        answer_txt_path = answer_dir_path / "answer.txt"
        answer_txt_path.write_text("Invalid input")

        # Score the problem
        score, errors, discussion = await benchmark.score_problem(
            problem=problem,
            agent_workdir=str(workdir_path),
            agent_answer_dir=str(answer_dir_path),
            container_name="test_container"
        )

        # Verify the scoring
        assert score == 0.0
        assert "Input validation failed" in errors
