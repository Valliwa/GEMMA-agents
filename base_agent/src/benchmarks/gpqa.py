# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GPQA (Graduate-Level Google-Proof Q&A) Benchmark.

This benchmark evaluates an agent's ability to answer graduate-level questions across
various domains that are designed to be "Google-proof" - meaning they require deep
understanding rather than just information lookup.

The benchmark uses the GPQA dataset from HuggingFace (idavidrein/gpqa), which contains
multiple choice questions with one correct answer and three incorrect answers. Questions
are presented with randomized answer choices, and responses must be in the format (A),
(B), (C), or (D).
"""

import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, ClassVar, List, Tuple
from dotenv import load_dotenv
from datasets import load_dataset

from .base import BaseBenchmark, Problem


@dataclass
class GPQAExample:
    """A single GPQA dataset example.

    Contains the question text, correct answer, and three incorrect answer choices.
    Each example has a unique problem ID for tracking purposes.
    """
    question: str
    correct_answer: str
    incorrect_answer1: str
    incorrect_answer2: str
    incorrect_answer3: str
    problem_id: str

    @classmethod
    def from_raw(cls, example: dict) -> "GPQAExample":
        """Create a GPQAExample from a raw dataset example.

        Args:
            example: Raw dictionary from the HuggingFace dataset

        Returns:
            GPQAExample with fields populated from the raw data
        """
        return cls(
            question=example["Question"],
            correct_answer=example["Correct Answer"],
            incorrect_answer1=example["Incorrect Answer 1"],
            incorrect_answer2=example["Incorrect Answer 2"],
            incorrect_answer3=example["Incorrect Answer 3"],
            problem_id=str(example.get("id", ""))
        )


class GPQABenchmark(BaseBenchmark):
    """Benchmark for the GPQA (Graduate-Level Google-Proof Q&A) dataset.

    This benchmark evaluates an agent's ability to answer graduate-level
    questions across various domains that require deep understanding rather
    than just information lookup.

    The questions are presented in multiple-choice format with randomized
    answer choices. Responses must be in the format (A), (B), (C), or (D).
    """

    name = "gpqa"

    def __init__(self, seed: int | None = 1, subset_size: int | None = 20):
        """Initialize the GPQA benchmark.

        Args:
            seed: Random seed for consistent answer shuffling across runs
            subset_size: Number of questions to sample (None for full dataset)

        Raises:
            ValueError: If subset_size is negative or larger than dataset
            RuntimeError: If HF_TOKEN is missing or dataset fails to load
        """
        super().__init__(seed, subset_size)

        # Validate inputs
        if subset_size is not None and subset_size <= 0:
            raise ValueError("subset_size must be positive")

        # Load HuggingFace token from environment
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise RuntimeError(
                "HF_TOKEN not found in environment. Please set it in .env file."
            )

        try:
            # Load dataset from HuggingFace with authentication
            dataset = load_dataset(
                "idavidrein/gpqa",
                "gpqa_diamond",  # Use the diamond GPQA configuration
                trust_remote_code=True,
                token=hf_token
            )
            self.test_data = [GPQAExample.from_raw(ex) for ex in dataset["train"]]

            # Validate subset size
            if subset_size is not None and subset_size > len(self.test_data):
                raise ValueError(
                    f"subset_size ({subset_size}) exceeds dataset size ({len(self.test_data)})"
                )

            # Create randomized subset if requested
            if subset_size is not None:
                random.seed(seed)
                self.test_data = random.sample(self.test_data, subset_size)

            # Convert to Problem instances with randomized answer choices
            self._data = []
            for i, ex in enumerate(self.test_data):
                # Create list of choices and randomize their order
                choices = [
                    ex.correct_answer,
                    ex.incorrect_answer1,
                    ex.incorrect_answer2,
                    ex.incorrect_answer3
                ]
                random.seed(seed + i)  # Use different seed for each example
                random.shuffle(choices)

                # Find index of correct answer
                correct_idx = choices.index(ex.correct_answer)

                # Create problem with randomized choices
                self._data.append(Problem(
                    problem_id=ex.problem_id or str(i),
                    statement=self._format_question(ex.question, choices, i + 1),
                    answer=f"({chr(65 + correct_idx)})",  # Convert to letter (A, B, C, D)
                    answer_discussion=None
                ))

        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPQA benchmark: {str(e)}")

    def _format_question(self, question: str, choices: List[str], problem_num: int) -> str:
        """Format question with choices in a clear, standardized format.

        Args:
            question: The question text
            choices: List of 4 answer choices in randomized order
            problem_num: Problem number for display purposes

        Returns:
            Formatted question string with problem number, clear instructions,
            and lettered answer choices
        """
        # Build formatted question with clear instructions
        result = [
            f"Problem {problem_num}:",
            "",
            question,
            "",
            "Instructions:",
            "- Choose the BEST answer from the options below",
            "- Respond with ONLY the letter choice in parentheses: (A), (B), (C), or (D)",
            "- Do not include any other text or explanation in your answer",
            "",
            "Answer Choices:"
        ]

        # Add formatted answer choices
        for i, choice in enumerate(choices):
            result.append(f"({chr(65 + i)}) {choice}")

        return "\n".join(result)

    @property
    def problems(self) -> list[Problem]:
        """Get the list of problems for this benchmark."""
        return self._data

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> Tuple[float, str | None, str | None]:
        """Score the answer from the agent against the ground truth.

        Args:
            problem: The problem being scored
            agent_workdir: Path to agent's working directory
            agent_answer_dir: Path to directory containing answer.txt
            container_name: Name of the agent's container

        Returns:
            Tuple of:
            - score (0.0 or 1.0)
            - error message if any
            - additional answer discussion if any

        The answer must be in the format (A), (B), (C), or (D) to be valid.
        """
        try:
            # Read and validate answer file exists
            answer_path = Path(agent_answer_dir) / "answer.txt"
            if not answer_path.exists():
                return 0.0, "No answer.txt file found", None

            # Read and clean answer
            agent_answer = answer_path.read_text().strip()

            # Check answer format
            if len(agent_answer) != 3:
                return 0.0, "Answer must be exactly 3 characters in format (A)", None

            if not (agent_answer.startswith("(") and agent_answer.endswith(")")):
                return 0.0, "Answer must be in format (A), (B), (C), or (D)", None

            letter = agent_answer[1]
            if letter not in "ABCD":
                return 0.0, "Answer letter must be one of A, B, C, or D", None

            # Compare with ground truth
            score = float(agent_answer == problem.answer)
            discussion = None if score == 1.0 else f"Incorrect. The correct answer was {problem.answer}."

            return score, None, discussion

        except Exception as e:
            return 0.0, f"Error scoring answer: {str(e)}", None


if __name__ == "__main__":
    import tempfile

    def run_test_case(benchmark: GPQABenchmark, answer_dir: Path,
                      ground_truth: str, agent_answer: str, should_pass: bool):
        """Helper function to run a single test case"""
        print(f"\nTESTING: '{ground_truth}' vs '{agent_answer}' (should_pass={should_pass})")

        problem = benchmark.problems[0]  # Use first problem as template
        problem.answer = ground_truth

        answer_file = answer_dir / "answer.txt"
        answer_file.write_text(agent_answer)

        score, error, _ = benchmark.score_problem(
            problem, str(answer_dir.parent), str(answer_dir), "test"
        )

        assert score == (1.0 if should_pass else 0.0), \
            f"Failed: '{ground_truth}' vs '{agent_answer}' got {score}, expected {1.0 if should_pass else 0.0}"

    # Create test environment
    benchmark = GPQABenchmark(seed=42, subset_size=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        answer_dir = Path(tmpdir) / "answers"
        answer_dir.mkdir()

        print("\nTesting answer format validation...")
        test_cases = [
            ("(A)", "(A)", True),
            ("(A)", "(B)", False),
            ("(A)", "A", False),
            ("(A)", "a", False),
            ("(A)", "(E)", False),
            ("(B)", "(B)", True),
            ("(C)", "(C)", True),
            ("(D)", "(D)", True),
            ("(A)", " (A) ", True),  # Should handle whitespace
            ("(A)", "(a)", False),  # Case sensitive
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        print("\nAll tests passed! ✨")
