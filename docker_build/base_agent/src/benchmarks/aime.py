# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import logging

from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass

from .base import BaseBenchmark, Problem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AIMEExample:
    """A single AIME example."""
    problem_id: str  # e.g., "2024-I-1"
    problem: str
    solution: str
    answer: int

    @classmethod
    def from_raw(cls, example: dict) -> "AIMEExample":
        """Create an AIMEExample from a raw dataset example."""
        return cls(
            problem_id=str(example["ID"]),
            problem=example["Problem"].strip(),
            solution=example["Solution"].strip(),
            answer=int(example["Answer"])  # AIME answers are always integers
        )


class AIMEBenchmark(BaseBenchmark):
    """Benchmark for the American Invitational Mathematics Examination (AIME) 2024 dataset.

    The AIME is a prestigious high school mathematics competition known for its challenging
    mathematical problems. All answers in AIME are integers.
    """

    name = "aime"

    def __init__(self, seed: int | None = 1, subset_size: int | None = 20):
        super().__init__(seed, subset_size)

        # Load dataset from HuggingFace
        dataset = load_dataset("Maxwell-Jia/AIME_2024")
        self.test_data = [AIMEExample.from_raw(ex) for ex in dataset["train"]]  # Dataset only has train split

        # Create randomized subset if requested
        if subset_size is not None:
            random.seed(seed)
            self.test_data = random.sample(self.test_data, subset_size)

        # Convert to Problem instances
        self._data = [
            Problem(
                problem_id=ex.problem_id,
                statement=ex.problem,
                answer=ex.answer,
                answer_discussion=ex.solution,
            )
            for ex in self.test_data
        ]

    @property
    def problems(self) -> list[Problem]:
        return self._data

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """Score the answer to the problem.

        Since AIME answers are always integers, we can do exact matching without
        any floating-point comparison.

        Returns:
            tuple of:
            - score (0.0 or 1.0)
            - error message (if any)
            - solution discussion
        """
        try:
            answer_path = Path(agent_answer_dir) / "answer.txt"
            llm_answer = answer_path.read_text().strip()

            # Clean the answer by removing any commas and whitespace
            llm_answer = llm_answer.replace(",", "").replace(" ", "")

            # Convert to integer and compare exactly
            try:
                answer_int = int(llm_answer)
                if answer_int == problem.answer:
                    return 1.0, None, problem.answer_discussion
                return 0.0, None, problem.answer_discussion
            except ValueError:
                return 0.0, "Answer must be an integer", problem.answer_discussion

        except Exception as e:
            logger.debug(f"Error in AIME scoring: {e}")
            return 0.0, str(e), problem.answer_discussion


if __name__ == "__main__":
    import tempfile

    def run_test_case(benchmark: AIMEBenchmark, answer_dir: Path,
                      ground_truth: int, agent_answer: str, should_pass: bool):
        """Helper function to run a single test case"""
        print(f"\nTESTING: '{ground_truth}' vs '{agent_answer}' (should_pass={should_pass})")

        # Use first problem as template but override answer
        problem = benchmark.problems[0]
        problem.answer = ground_truth
        problem.answer_discussion = "Test discussion"

        answer_file = answer_dir / "answer.txt"
        answer_file.write_text(agent_answer)

        score, error, _ = benchmark.score_problem(
            problem, str(answer_dir.parent), str(answer_dir), "test"
        )

        assert score == (1.0 if should_pass else 0.0), \
            f"Failed: '{ground_truth}' vs '{agent_answer}' got {score}, expected {1.0 if should_pass else 0.0}"
        if error:
            print(f"Error message: {error}")

    # Create test environment
    benchmark = AIMEBenchmark()

    with tempfile.TemporaryDirectory() as tmpdir:
        answer_dir = Path(tmpdir) / "answers"
        answer_dir.mkdir()

        print("\nTesting basic integer answers...")
        test_cases = [
            (42, "42", True),
            (42, "42.0", False),  # Must be exact integer
            (1000, "1,000", True),  # Allow commas
            (1000, "1000", True),
            (1000, " 1000 ", True),  # Allow whitespace
            (42, "abc", False),  # Non-numeric
            (-123, "-123", True),  # Negative numbers
            (0, "0", True),
            (0, "0.0", False),
            (42, "41", False),  # Wrong answer
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        # Test that the dataset loads correctly
        print("\nTesting dataset loading...")
        assert len(benchmark.problems) > 0, "Dataset should not be empty"
        assert all(isinstance(p.answer, int) for p in benchmark.problems), \
            "All answers should be integers"
        assert all(isinstance(p.problem_id, str) for p in benchmark.problems), \
            "All problem IDs should be strings"
        assert all(p.problem_id.startswith("2024-") for p in benchmark.problems), \
            "All problem IDs should start with 2024-"

        # Test subset functionality
        print("\nTesting subset functionality...")
        subset_size = 5
        benchmark_subset = AIMEBenchmark(seed=42, subset_size=subset_size)
        assert len(benchmark_subset.problems) == subset_size, \
            f"Subset size should be {subset_size}, got {len(benchmark_subset.problems)}"

        # Test seed reproducibility
        print("\nTesting seed reproducibility...")
        benchmark_subset1 = AIMEBenchmark(seed=42, subset_size=subset_size)
        benchmark_subset2 = AIMEBenchmark(seed=42, subset_size=subset_size)
        assert [p.problem_id for p in benchmark_subset1.problems] == \
               [p.problem_id for p in benchmark_subset2.problems], \
            "Same seed should produce same subset"

        # Test different seeds produce different subsets
        benchmark_subset3 = AIMEBenchmark(seed=43, subset_size=subset_size)
        assert [p.problem_id for p in benchmark_subset1.problems] != \
               [p.problem_id for p in benchmark_subset3.problems], \
            "Different seeds should produce different subsets"

        print("\nAll tests passed! ✨")
