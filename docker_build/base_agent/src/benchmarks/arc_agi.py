# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import logging

from pathlib import Path
from typing import List, Dict, Any
from datasets import load_dataset
from dataclasses import dataclass

from .base import BaseBenchmark, Problem

logger = logging.getLogger(__name__)

def format_grid(grid: List[List[int]]) -> str:
    return "\n".join([" ".join(map(str, row)) for row in grid])


@dataclass
class ARCExample:
    """A single ARC AGI example"""

    train_inputs: List[List[List[int]]]
    train_outputs: List[List[List[int]]]
    test_input: List[List[int]]
    test_output: List[List[int]]

    @classmethod
    def from_raw(cls, example: Dict[str, Any]) -> "ARCExample":
        """Create an ARCExample from a raw dataset example"""
        # Each example has a single test case
        return cls(
            train_inputs=[eg['input'] for eg in example["train"]],
            train_outputs=[eg['output'] for eg in example["train"]],
            test_input=example["test"][0]['input'],
            test_output=example["test"][0]['output'],
        )

    def __str__(self) -> str:
        """Convert the example to a string format for the problem statement"""

        string = "You are solving a pattern recognition task. You will be presented with a few examples to demonstrate the pattern first.\n"
        for i, (eg_in, eg_out) in enumerate(zip(self.train_inputs, self.train_outputs)):
            string += (
                f"Example {i}: When the input is:\n\n{format_grid(eg_in)}\n\n"
                f"the output is:\n\n{format_grid(eg_out)}\n\n"
            )

        return string + (
            f"End of examples. Test case:\nWhen the input is:\n\n"
            f"{format_grid(self.test_input)}\n\n"
            f"What is the output?\n"
            f"Express your answer as just the grid of numbers in the same format with no additional text."
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {"train_inputs": self.train_inputs, "train_outputs": self.train_outputs,
                "test_input": self.test_input, "test_output": self.test_output}

    @classmethod
    def from_dict(cls, data: Dict) -> "ARCExample":
        """Create from dictionary after deserialization"""
        return cls(train_inputs=data["train_inputs"], train_outputs=data["train_outputs"],
                   test_input=data["test_input"], test_output=data["test_output"])


class ARCAGIBenchmark(BaseBenchmark):
    """
    ARC AGI (Abstraction and Reasoning Corpus) benchmark implementation.

    This benchmark evaluates an agent's ability to infer abstract patterns and
    apply them to transform input grids. Each problem consists of:
    - An input grid of integers
    - A target output grid

    The agent must determine the transformation pattern and provide the correct output.
    """

    name = "arc_agi"

    def __init__(self, seed: int | None = None, subset_size: int | None = None):
        """Initialize the ARC AGI benchmark.

        Args:
            split: Dataset split to use ("trial", "training", or "evaluation")
            max_problems: Optional maximum number of problems to load
        """
        super().__init__(seed, subset_size)
        split = "evaluation"

        # Validate inputs
        if subset_size is not None and subset_size <= 0:
            raise ValueError("subset_size must be positive")

        try:
            # Load ARC AGI dataset from HuggingFace
            dataset = load_dataset("lordspline/arc-agi", split=split)

            # Convert examples to problems
            self._data = []
            for i, example in enumerate(dataset):
                try:
                    arc_example = ARCExample.from_raw(example)
                    self._data.append(
                        Problem(
                            problem_id=str(i),
                            statement=str(arc_example),
                            answer=arc_example.to_dict(),
                            answer_discussion=None,
                        )
                    )

                except Exception as e:
                    logger.warning(f"Failed to process example {i}: {str(e)}")
                    continue

            # Create randomized subset if requested
            if subset_size is not None:
                random.seed(seed)
                self._data = random.sample(self._data, subset_size)

        except Exception as e:
            logger.error(f"Error loading ARC AGI dataset: {e}")
            self._data = [
                Problem(
                    problem_id="error",
                    statement="Error loading ARC AGI dataset",
                    answer={"error": str(e)},
                    answer_discussion=None,
                )
            ]

    @property
    def problems(self) -> list[Problem]:
        """Return list of ARC AGI problems"""
        return self._data

    def parse_grid(self, text: str) -> List[List[int]]:
        """Parse a text grid into a 2D list of integers.

        Args:
            text: Grid text with space-separated numbers and newlines between rows

        Returns:
            2D list of integers representing the grid

        Raises:
            ValueError: If the grid cannot be parsed or is invalid
        """
        try:
            # Split into rows and parse numbers
            rows = text.strip().split("\n")
            grid = []
            width = None

            for row in rows:
                # Convert each row to integers
                numbers = [int(x) for x in row.strip().split()]

                # Validate consistent width
                if width is None:
                    width = len(numbers)
                elif len(numbers) != width:
                    raise ValueError("Inconsistent row lengths in grid")

                grid.append(numbers)

            return grid

        except Exception as e:
            raise ValueError(f"Failed to parse grid: {str(e)}")

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """Score a submitted answer against the reference solution.

        The answer must be an exact match with the reference grid, accounting for:
        - Grid dimensions
        - All numeric values
        - Spacing and formatting is flexible

        Args:
            problem: Problem instance containing the reference answer
            agent_workdir: The agent's working directory where any files would have been written
            agent_answer_dir: Where the agent should have written its answer.txt
            container_name: The name of the running docker container where the agent run

        Returns:
            1.0 for exact match, 0.0 otherwise
        """
        try:
            answer_path = Path(agent_answer_dir) / "answer.txt"
            llm_answer = answer_path.read_text().strip()

            # Parse submitted answer into grid
            submitted_grid = self.parse_grid(llm_answer)

            # Get reference output grid
            reference = ARCExample.from_dict(problem.answer)
            reference_grid = reference.test_output

            # Compare dimensions
            if len(submitted_grid) != len(reference_grid) or len(
                submitted_grid[0]
            ) != len(reference_grid[0]):
                return 0.0, None, None

            numel = 0
            correct = 0
            for i in range(len(reference_grid)):
                for j in range(len(reference_grid[0])):
                    numel += 1
                    correct += 1 if submitted_grid[i][j] == reference_grid[i][j] else 0

            proportion = correct / numel * 100

            # Compare all values
            return (
                float(
                    all(
                        submitted_grid[i][j] == reference_grid[i][j]
                        for i in range(len(reference_grid))
                        for j in range(len(reference_grid[0]))
                    )
                ),
                None,
                f"The proportion of correct grid entries was {proportion:.2f}%",
            )

        except Exception as e:
            logger.debug(f"Error scoring ARC AGI answer: {e}")
            return 0.0, str(e), None


if __name__ == '__main__':
    import os

    bench = ARCAGIBenchmark()
    fst = bench.problems[0]

    # Create a temporary file and get its path
    try:
        os.remove("/tmp/answer.txt")
    except Exception:
        pass

    # Write to the temporary file
    with open("/tmp/answer.txt", "w") as f:
        grid = ARCExample.from_dict(fst.answer).test_output
        print(grid)
        submission = []
        for i in range(len(grid)):
            submission.append([])
            for j in range(len(grid[0])):
                submission[i].append(0)
        print(submission)
        f.write(format_grid(submission))

    score, _, additional = bench.score_problem(fst, "/tmp", "/tmp", "")
    print(score, additional)

    os.remove("/tmp/answer.txt")
