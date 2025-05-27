# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import shutil
import logging

from typing import Literal
from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass

from .base import BaseBenchmark, Problem

logger = logging.getLogger(__name__)


@dataclass
class HumanEvalProblem(Problem):
    """Problem subclass specifically for HumanEval tasks"""

    prompt: str
    test: str
    entry_point: str
    canonical_solution: str

    @classmethod
    def from_raw(cls, example: dict) -> "HumanEvalProblem":
        """Create a HumanEvalProblem from a raw dataset example."""
        return cls(
            problem_id=example["task_id"],
            statement=(
                f"Please complete the following Python function:\n\n{example['prompt']}"
                f"\n\nReturn as your answer the full function `{example['entry_point']}`."
                if example["task_id"] != "error"
                else "Error loading HumanEval dataset"
            ),
            answer=example["canonical_solution"],
            answer_discussion=example["canonical_solution"],
            prompt=example["prompt"],
            test=example["test"],
            entry_point=example["entry_point"],
            canonical_solution=example["canonical_solution"],
        )


class HumanEvalBenchmark(BaseBenchmark):
    """Benchmark for evaluating code generation using the HumanEval dataset"""

    name = "humaneval"

    def __init__(self, seed: int = 1, subset_size: int | None = 8):
        super().__init__(seed, subset_size)
        mode: Literal["openai", "file"] = "file"
        self.mode = mode
        self.seed = seed
        self.subset_size = subset_size
        if mode not in ["openai", "file"]:
            raise ValueError("Mode must be either 'openai' or 'file'")
        try:
            # Load HumanEval dataset from HuggingFace
            dataset = load_dataset("openai/openai_humaneval")
            self._all_problems = [
                HumanEvalProblem.from_raw(ex) for ex in dataset["test"]
            ]
        except Exception as e:
            logger.error(f"Error loading HumanEval dataset: {e}")
            self._all_problems = [
                Problem(
                    problem_id="error",
                    statement="Error loading HumanEval dataset",
                    answer=str(e),
                    answer_discussion=None,
                )
            ]

        self._problems_shuffled = False
        self._data = None

    def _shuffle_problems(self) -> None:
        """Shuffle the problems list using the specified seed"""
        if not self._problems_shuffled:
            all_problems = self._all_problems.copy()
            random.seed(self.seed)
            random.shuffle(all_problems)
            self._problems_shuffled = True
            # Apply subset if specified
            if self.subset_size is not None:
                self._data = all_problems[: min(self.subset_size, len(all_problems))]
            else:
                self._data = all_problems

    @property
    def problems(self) -> list[Problem]:
        if self._data is None:
            self._shuffle_problems()
        return self._data

    async def setup_problem(
        self, problem: Problem, problem_data_dir: Path, container_name: str
    ) -> None:
        """
        Setup the test environment for this problem based on the mode.
        - "openai" mode: No setup needed as agent submits complete function
        - "file" mode: Creates separate problem.py and tests.py files
        """
        if not isinstance(problem, HumanEvalProblem):
            return

        if self.mode == "file":
            # Create problem.py with just the function signature
            problem_file = problem_data_dir / "problem.py"
            problem_content = f"{problem.prompt}\n# Write your implementation here\n"
            problem_file.write_text(problem_content)

            # Create tests.py with imports and test code
            test_file = problem_data_dir / "tests.py"
            test_content = f"""from problem import {problem.entry_point}

{problem.test}

if __name__ == "__main__":
    try:
        check({problem.entry_point})
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {{e}}")
    except Exception as e:
        print(f"ERROR: {{e}}")
"""
            test_file.write_text(test_content)

            # Modify the problem statement for file mode
            file_mode_statement = (
                f"Complete the function implementation in 'problem.py'. "
                f"The file contains the function signature and docstring. "
                f"Your implementation must pass all tests in 'tests.py'."
            )
            setattr(problem, "statement", file_mode_statement)

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """Score the submitted answer for the HumanEval problem based on the mode."""
        try:
            if not isinstance(problem, HumanEvalProblem):
                return 0.0, "Invalid problem type", None

            workdir_path = Path(agent_workdir)
            answer_dir_path = Path(agent_answer_dir)

            if self.mode == "openai":
                # Get the agent's answer
                answer_path = answer_dir_path / "answer.txt"
                if not answer_path.exists():
                    return 0.0, "No answer file found", None
                llm_answer = answer_path.read_text().strip()

                # Create problem.py with the solution
                problem_file = workdir_path / "problem.py"
                problem_file.write_text(llm_answer)

                # Create tests.py that imports the solution
                test_file = workdir_path / "tests.py"
                test_content = f"""from problem import {problem.entry_point}

{problem.test}

if __name__ == "__main__":
    try:
        check({problem.entry_point})
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {{e}}")
    except Exception as e:
        print(f"ERROR: {{e}}")
"""
                test_file.write_text(test_content)
            else:  # file mode
                # problem.py should already exist and be edited by the agent
                problem_file = workdir_path / "problem.py"
                if not problem_file.exists():
                    return 0.0, "Problem file not found or not edited", None

                # Overwrite tests.py to prevent tampering
                test_file = workdir_path / "tests.py"
                test_content = f"""from problem import {problem.entry_point}

{problem.test}

if __name__ == "__main__":
    try:
        check({problem.entry_point})
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED: {{e}}")
    except Exception as e:
        print(f"ERROR: {{e}}")
"""
                test_file.write_text(test_content)

            # Run the tests in the container
            import subprocess

            result = subprocess.run(
                [
                    "docker",
                    "exec",
                    container_name,
                    "python",
                    "/home/agent/workdir/tests.py",
                ],
                capture_output=True,
                text=True,
            )

            # Copy the solution to humaneval_answer.py in the answer directory
            shutil.copy(
                workdir_path / "problem.py", answer_dir_path / "humaneval_answer.py"
            )

            # Count the number of test cases
            test_lines = problem.test.strip().split("\n")
            test_count = sum(
                1 for line in test_lines if line.strip().startswith("assert")
            )

            # Calculate score based on proportion of tests passed
            if test_count == 0:
                return 0.0, "No test cases found", None

            if "ERROR" in result.stdout:
                return 0.0, f"Error running tests: {result.stdout}", None

            if "PASSED" in result.stdout:
                return 1.0, None, None

            # Count failed tests
            failed_count = result.stdout.count("FAILED")
            if failed_count > test_count:
                failed_count = test_count

            # Calculate proportion of tests passed
            passed_count = test_count - failed_count
            score = passed_count / test_count

            return score, f"{passed_count}/{test_count} tests passed", None

        except Exception as e:
            logger.error(f"Error scoring HumanEval problem: {e}")
            return 0.0, str(e), None


# Test entrypoint
if __name__ == "__main__":
    # Running:
    import asyncio

    async def main():
        benchmark = HumanEvalBenchmark(mode="openai")  # or mode="file"
        problems = benchmark.problems
        print(f"Loaded {len(problems)} problems from HumanEval dataset")
        if problems:
            print("\nExample problem:")
            example = problems[0]
            print(f"Problem ID: {example.problem_id}")
            print(f"Statement:\n{example.statement}")
            if isinstance(example, HumanEvalProblem):
                print(f"\nEntry point: {example.entry_point}")
                print(f"Test:\n{example.test}")

    asyncio.run(main())
