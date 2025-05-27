# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the LiveCode benchmark implementation."""
import pytest
import tempfile
import json
import shutil
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock, mock_open

from src.benchmarks.livecodebench import (
    LiveCodeBenchmark,
    LiveCodeBenchProblem,
    Platform,
    Difficulty,
    TestType,
    Test,
    IMPORT_STRING,
)


class TestTest:
    """Tests for the Test class that represents test cases."""

    def test_from_stdin(self):
        """Test creation of stdin test types."""
        test = Test(
            input="1 2 3\n",
            output="6\n",
            testtype=TestType.STDIN
        )

        assert test.testtype == TestType.STDIN
        assert test.input == "1 2 3\n"
        assert test.output == "6\n"

    def test_from_functional(self):
        """Test creation of functional test types."""
        test = Test(
            input='[1, 2, 3]',
            output='6',
            testtype=TestType.FUNCTIONAL
        )

        assert test.testtype == TestType.FUNCTIONAL
        assert isinstance(test.input, str)
        assert isinstance(test.output, str)

    def test_json_conversion(self):
        """Test JSON conversion of test inputs/outputs."""
        test = Test(
            input={"nums": [1, 2, 3]},
            output={"result": 6},
            testtype=TestType.FUNCTIONAL
        )

        assert test.input == '{"nums": [1, 2, 3]}'
        assert test.output == '{"result": 6}'

    def test_format_example_stdin(self):
        """Test formatting of stdin test examples."""
        test = Test(
            input="5\n1 2 3 4 5\n",
            output="15\n",
            testtype=TestType.STDIN
        )

        example = test.format_example()
        assert "Sample Input:" in example
        assert "5\n1 2 3 4 5" in example
        assert "Sample Output:" in example
        assert "15" in example

    def test_format_example_functional(self):
        """Test formatting of functional test examples."""
        test = Test(
            input='[1, 2, 3]',
            output='6',
            testtype=TestType.FUNCTIONAL
        )

        example = test.format_example()
        assert "Sample Call:" in example
        assert "solution(" in example
        assert "Expected Output:" in example


class TestLiveCodeBenchProblem:
    """Tests for the LiveCodeBenchProblem class."""

    @pytest.fixture
    def example_data(self):
        """Create example problem data for testing."""
        return {
            "question_id": "test_123",
            "question_title": "Sum of Numbers",
            "question_content": "Given a list of numbers, return their sum.",
            "platform": "leetcode",
            "contest_id": "weekly-123",
            "contest_date": "2025-01-01T00:00:00",
            "difficulty": "easy",
            "public_test_cases": [
                {
                    "input": "[1, 2, 3]",
                    "output": "6",
                    "testtype": "functional"
                }
            ],
            "private_test_cases": [
                {
                    "input": "[4, 5, 6]",
                    "output": "15",
                    "testtype": "functional"
                }
            ],
            "metadata": "{}"
        }

    def test_from_raw(self, example_data):
        """Test problem creation from raw data."""
        problem = LiveCodeBenchProblem.from_raw(example_data)

        assert problem.problem_id == "test_123"
        assert problem.question_title == "Sum of Numbers"
        assert problem.platform == Platform.LEETCODE
        assert problem.difficulty == Difficulty.EASY
        assert len(problem.public_test_cases) == 1
        assert len(problem.private_test_cases) == 1
        assert problem.starter_code.startswith("def solution")

    def test_create_test_file(self, example_data):
        """Test creation of test files for individual test cases."""
        problem = LiveCodeBenchProblem.from_raw(example_data)
        test = problem.public_test_cases[0]

        test_content = problem.create_test(1, test)

        assert "from solution import solution" in test_content
        assert "def run_test():" in test_content
        assert "if __name__ == '__main__':" in test_content

    @pytest.mark.asyncio
    async def test_score_problem(self, example_data):
        """Test scoring a correct solution."""
        problem = LiveCodeBenchProblem.from_raw(example_data)

        solution_code = """
def solution(nums):
    return sum(nums)
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Create solution file
            solution_file = tmp_path / "solution.py"
            solution_file.write_text(solution_code)

            # Score the solution
            with patch("subprocess.run") as mock_run:
                # Mock successful test execution
                mock_run.return_value = MagicMock(
                    stdout="PASSED\n",
                    stderr="",
                )

                score, errors, _ = await problem.score_problem(
                    problem=problem,
                    agent_workdir=str(tmp_path),
                    agent_answer_dir="/fake/answer_dir",
                    container_name="test_container"
                )

                assert score == 1.0
                assert errors is None

    @pytest.mark.asyncio
    async def test_error_handling(self, example_data):
        """Test handling of various error conditions."""
        problem = LiveCodeBenchProblem.from_raw(example_data)

        # Test missing solution file
        with tempfile.TemporaryDirectory() as tmp_dir:
            score, errors, _ = await problem.score_problem(
                problem=problem,
                agent_workdir=tmp_dir,
                agent_answer_dir="/fake/answer_dir",
                container_name="test_container"
            )

            assert score == 0.0
            assert "No solution.py file found" in errors


@pytest.mark.slow
def test_benchmark_initialization():
    """
    Test initialization of the LiveCode benchmark with actual dataset.
    Note: This test loads the real dataset and may take a while / use significant memory.
    Run with --run-slow flag to include this test.
    """
    benchmark = None
    try:
        # Initialize with very small subset for testing
        benchmark = LiveCodeBenchmark(seed=42, subset_size=3)

        print("\nLoading dataset, this may take a few minutes...")

        # Basic integrity checks
        assert benchmark.name == "livecodebench"
        assert len(benchmark.problems) == 3

        # Verify problems have expected structure
        for problem in benchmark.problems:
            assert isinstance(problem, LiveCodeBenchProblem)
            assert isinstance(problem.platform, Platform)
            assert isinstance(problem.difficulty, Difficulty)
            assert hasattr(problem, "public_test_cases")
            assert len(problem.public_test_cases) > 0

            # Check test case structure
            for test in problem.public_test_cases:
                assert isinstance(test, Test)
                assert isinstance(test.input, str)
                assert isinstance(test.output, str)
                assert isinstance(test.testtype, TestType)

        print("Dataset loaded and verified successfully.")

    finally:
        # Clean up any dataset files
        if benchmark is not None and hasattr(benchmark, "_dataset_path"):
            dataset_dir = benchmark._dataset_path.parent
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)


@pytest.mark.skip(reason="Takes too long and requires significant memory")
def test_full_benchmark_initialization():
    """Load the full benchmark dataset as a stress test."""
    benchmark = None
    try:
        # Initialize with full dataset
        benchmark = LiveCodeBenchmark(seed=42, subset_size=None)

        print("\nLoading full dataset, this will take several minutes...")

        # Basic integrity checks
        assert benchmark.name == "livecodebench"
        assert len(benchmark.problems) > 100  # Should have many problems

        # Check a few random problems
        import random
        random.seed(42)
        for _ in range(5):
            problem = random.choice(benchmark.problems)
            assert isinstance(problem, LiveCodeBenchProblem)
            assert isinstance(problem, LiveCodeBenchProblem)
            assert isinstance(problem.platform, Platform)
            assert isinstance(problem.difficulty, Difficulty)
            assert hasattr(problem, "public_test_cases")
            assert len(problem.public_test_cases) > 0

        print("Full dataset loaded and verified successfully.")

    finally:
        # Clean up dataset files
        if benchmark is not None and hasattr(benchmark, "_dataset_path"):
            dataset_dir = benchmark._dataset_path.parent
            if dataset_dir.exists():
                shutil.rmtree(dataset_dir)


@pytest.mark.asyncio
async def test_setup_problem():
    """Test problem setup functionality."""
    # Create a test problem
    problem = LiveCodeBenchProblem.from_raw({
        "question_id": "test_123",
        "question_title": "Test Problem",
        "question_content": "Test content",
        "platform": "leetcode",
        "contest_id": "contest_1",
        "contest_date": "2025-01-01T00:00:00",
        "difficulty": "easy",
        "public_test_cases": [
            {
                "input": "[1, 2, 3]",
                "output": "6",
                "testtype": "functional"
            }
        ],
        "private_test_cases": [],
        "metadata": "{}"
    })

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        benchmark = LiveCodeBenchmark()

        # Setup the problem
        await benchmark.setup_problem(
            problem=problem,
            problem_data_dir=tmp_path,
            container_name="test_container"
        )

        # Verify solution.py was created
        solution_file = tmp_path / "solution.py"
        assert solution_file.exists()

        content = solution_file.read_text()
        assert IMPORT_STRING in content
        assert problem.starter_code in content


@patch("subprocess.run")
def test_solution_scoring(mock_run):
    """Test scoring solutions with different outcomes."""
    # Create a test benchmark
    benchmark = LiveCodeBenchmark()

    # Create a test problem
    problem = LiveCodeBenchProblem.from_raw({
        "question_id": "test_123",
        "question_title": "Test Problem",
        "question_content": "Test content",
        "platform": "leetcode",
        "contest_id": "contest_1",
        "contest_date": "2025-01-01T00:00:00",
        "difficulty": "easy",
        "public_test_cases": [
            {
                "input": "[1, 2, 3]",
                "output": "6",
                "testtype": "functional"
            },
            {
                "input": "[4, 5, 6]",
                "output": "15",
                "testtype": "functional"
            }
        ],
        "private_test_cases": [],
        "metadata": "{}"
    })

    # Create a temporary directory with a solution
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        solution_file = tmp_path / "solution.py"
        solution_file.write_text("def solution(nums): return sum(nums)")

        # Test passing all test cases
        mock_run.return_value = MagicMock(stdout="PASSED\n", stderr="")
        score, errors, _ = benchmark.score_problem(
            problem=problem,
            agent_workdir=str(tmp_path),
            agent_answer_dir="/fake/answer_dir",
            container_name="test_container"
        )
        assert score == 1.0
        assert errors is None

        # Test failing some test cases
        mock_run.side_effect = [
            MagicMock(stdout="PASSED\n", stderr=""),
            MagicMock(stdout="FAILED\n", stderr=""),
        ]
        score, errors, _ = benchmark.score_problem(
            problem=problem,
            agent_workdir=str(tmp_path),
            agent_answer_dir="/fake/answer_dir",
            container_name="test_container"
        )
        assert score == 0.5
        assert errors is not None

        # Test runtime error
        error_msg = "Runtime Error: division by zero"
        mock_run.return_value = MagicMock(
            stdout="",
            stderr=error_msg,
            returncode=1  # Non-zero return code indicates error
        )
        score, errors, _ = benchmark.score_problem(
            problem=problem,
            agent_workdir=str(tmp_path),
            agent_answer_dir="/fake/answer_dir",
            container_name="test_container"
        )
        assert score == 0.0
        assert error_msg in errors
