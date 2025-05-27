# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the GSM8K benchmark implementation."""
import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.benchmarks.gsm8k import GSM8KBenchmark, GSM8KExample


class TestGSM8KExample:
    """Tests for the GSM8KExample class."""

    def test_from_raw(self):
        """Test conversion from raw dataset example."""
        # Create a mock raw example
        raw_example = {
            "question": "John has 5 apples. He buys 2 more. How many does he have now?",
            "answer": "John has 5 apples initially.\nHe buys 2 more apples.\nSo he has 5 + 2 = <<5+2=7>> apples in total.\n#### 7"
        }

        example = GSM8KExample.from_raw(raw_example)

        assert example.answer == raw_example["answer"]
        assert example.steps == [
            "John has 5 apples initially.",
            "He buys 2 more apples.",
            "So he has 5 + 2 = <<5+2=7>> apples in total."
        ]
        assert example.final_answer == 7.0

    def test_extract_calculations(self):
        """Test extraction of calculations from solution steps."""
        raw_example = {
            "question": "Calculation test",
            "answer": "Step 1: Calculate 2 + 3 = <<2+3=5>>\nStep 2: Multiply by 4: 5 × 4 = <<5*4=20>>\n#### 20"
        }

        example = GSM8KExample.from_raw(raw_example)
        calculations = example.extract_calculations()

        assert len(calculations) == 2

        # First calculation
        expr1, expected1, actual1 = calculations[0]
        assert expr1 == "2+3"
        assert expected1 == 5
        assert actual1 == 5

        # Second calculation
        expr2, expected2, actual2 = calculations[1]
        assert expr2 == "5*4"
        assert expected2 == 20
        assert actual2 == 20


@pytest.mark.parametrize("subset_size", [None, 5, 10])
def test_benchmark_initialization(subset_size):
    """Test initializing the GSM8K benchmark with various subset sizes."""
    with patch("src.benchmarks.gsm8k.load_dataset") as mock_load_dataset:
        # Mock the dataset loading
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = [
            {"question": f"Question {i}", "answer": f"Some steps\n#### {i}"}
            for i in range(1, 21)  # Create 20 mock examples
        ]
        mock_load_dataset.return_value = mock_dataset

        benchmark = GSM8KBenchmark(seed=42, subset_size=subset_size)

        # Check benchmark properties
        assert benchmark.name == "gsm8k"

        # Verify subset_size is respected
        if subset_size:
            assert len(benchmark.problems) == subset_size
        else:
            assert len(benchmark.problems) == 20  # All examples

        # Verify problems have the expected structure
        for problem in benchmark.problems:
            assert isinstance(problem.statement, str)
            assert isinstance(problem.problem_id, str)  # Just check it's a string
            assert isinstance(problem.answer, float)
            assert isinstance(problem.answer_discussion, str)


@pytest.mark.asyncio
async def test_score_problem_correct():
    """Test scoring a correct GSM8K answer."""
    with patch("src.benchmarks.gsm8k.load_dataset") as mock_load_dataset:
        # Mock the dataset loading
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = [
            {"question": "Question 1", "answer": "Some steps\n#### 42"}
        ]
        mock_load_dataset.return_value = mock_dataset

        benchmark = GSM8KBenchmark(seed=42, subset_size=1)
        problem = benchmark.problems[0]

        # Create a temporary directory for the answer
        with tempfile.TemporaryDirectory() as tmp_dir:
            answer_dir = Path(tmp_dir)

            # Create answer.txt with the correct answer
            answer_file = answer_dir / "answer.txt"
            answer_file.write_text("42")

            # Score the answer
            score, errors, discussion = await benchmark.score_problem(
                problem=problem,
                agent_workdir="/fake/workdir",
                agent_answer_dir=str(answer_dir),
                container_name="fake_container"
            )

            # Verify the scoring
            assert score == 1.0
            assert errors is None
            assert discussion is not None


@pytest.mark.asyncio
async def test_score_problem_incorrect():
    """Test scoring an incorrect GSM8K answer."""
    with patch("src.benchmarks.gsm8k.load_dataset") as mock_load_dataset:
        # Mock the dataset loading
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = [
            {"question": "Question 1", "answer": "Some steps\n#### 42"}
        ]
        mock_load_dataset.return_value = mock_dataset

        benchmark = GSM8KBenchmark(seed=42, subset_size=1)
        problem = benchmark.problems[0]

        # Create a temporary directory for the answer
        with tempfile.TemporaryDirectory() as tmp_dir:
            answer_dir = Path(tmp_dir)

            # Create answer.txt with an incorrect answer
            answer_file = answer_dir / "answer.txt"
            answer_file.write_text("43")

            # Score the answer
            score, errors, discussion = await benchmark.score_problem(
                problem=problem,
                agent_workdir="/fake/workdir",
                agent_answer_dir=str(answer_dir),
                container_name="fake_container"
            )

            # Verify the scoring
            assert score == 0.0
            assert errors is None
            assert discussion is not None


@pytest.mark.asyncio
async def test_score_problem_invalid_format():
    """Test scoring a GSM8K answer with invalid format."""
    with patch("src.benchmarks.gsm8k.load_dataset") as mock_load_dataset:
        # Mock the dataset loading
        mock_dataset = MagicMock()
        mock_dataset.__getitem__.return_value = [
            {"question": "Question 1", "answer": "Some steps\n#### 42"}
        ]
        mock_load_dataset.return_value = mock_dataset

        benchmark = GSM8KBenchmark(seed=42, subset_size=1)
        problem = benchmark.problems[0]

        # Create a temporary directory for the answer
        with tempfile.TemporaryDirectory() as tmp_dir:
            answer_dir = Path(tmp_dir)

            # Create answer.txt with an incorrectly formatted answer
            answer_file = answer_dir / "answer.txt"
            answer_file.write_text("The answer is forty-two")

            # Score the answer
            score, errors, discussion = await benchmark.score_problem(
                problem=problem,
                agent_workdir="/fake/workdir",
                agent_answer_dir=str(answer_dir),
                container_name="fake_container"
            )

            # Verify the scoring
            assert score == 0.0
            assert errors is not None  # Should have parsing errors
            assert "could not convert string to float" in errors.lower() or "invalid literal" in errors.lower()
