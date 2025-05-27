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

logger = logging.getLogger(__name__)


@dataclass
class GSMICExample:
    """A single GSM-IC example with irrelevant context."""

    question: str
    answer: float
    n_steps: int

    @classmethod
    def from_raw(cls, example: dict) -> "GSMICExample":
        """Create a GSMICExample from a raw dataset example."""
        return cls(
            question=example["question"].strip(),
            answer=float(str(example["answer"]).strip().replace(",", "")),
            n_steps=int(example["n_steps"]),
        )


class GSMICBenchmark(BaseBenchmark):
    """Benchmark for the GSM-IC dataset that tests mathematical reasoning with irrelevant context."""

    name = "gsm_ic"

    def __init__(self, seed: int | None = None, subset_size: int | None = None):
        """Initialize the GSM-IC benchmark.

        Args:
            subset_size: Number of problems to use (default 50 to match GSM8K implementation)
        """
        super().__init__(seed, subset_size)

        # Validate inputs
        if subset_size is not None and subset_size <= 0:
            raise ValueError("subset_size must be positive")

        # Load the dataset
        dataset = load_dataset("voidful/GSM-IC")
        self.data = [GSMICExample.from_raw(ex) for ex in dataset["validation"]]

        # Create problem instances, limiting to subset_size
        self._problems = [
            Problem(problem_id=str(i), statement=p.question, answer=p.answer, answer_discussion=None)
            for i, p in enumerate(self.data)
        ]

        # Create randomized subset if requested
        if subset_size is not None:
            random.seed(seed)
            self._problems = random.sample(self._problems, subset_size)

    @property
    def problems(self) -> list[Problem]:
        """Return the list of problems."""
        return self._problems

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """Score an answer from the LLM against the ground truth.

        Args:
            problem: Problem instance containing the ground truth
            llm_answer: Answer string from the LLM

        Returns:
            1.0 if the answer is correct, 0.0 otherwise
        """
        try:
            answer_path = Path(agent_answer_dir) / "answer.txt"
            llm_answer = answer_path.read_text().strip()

            # Clean and convert llm answer to float
            float_answer = float(llm_answer.strip().replace(",", "").replace(" ", ""))

            # Compare with small tolerance
            if abs(problem.answer - float_answer) < 1e-7:
                return 1.0, None, None
            return 0.0, None, None

        except Exception as e:
            logger.debug(f"Error in GSM-IC scoring: {e}")
            return 0.0, str(e), None
