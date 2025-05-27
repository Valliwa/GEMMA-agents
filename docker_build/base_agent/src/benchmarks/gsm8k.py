# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import random
import logging

from typing import List
from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass

from .base import BaseBenchmark, Problem

logger = logging.getLogger(__name__)


@dataclass
class GSM8KExample:
    """A single GSM8K example."""

    question: str
    answer: str
    steps: list[str]
    final_answer: float

    @classmethod
    def from_raw(cls, example: dict) -> "GSM8KExample":
        """Create a GSM8KExample from a raw dataset example."""
        # Split answer into steps and final answer
        answer_parts = example["answer"].split("####")
        steps = [s.strip() for s in answer_parts[0].split("\n") if s.strip()]
        final_answer = float(answer_parts[1].strip().replace(",", ""))

        return cls(
            question=example["question"].strip() + "\n\nWhen submitting your answer, please just give a single number with no accompanying text, units or other markings.",
            answer=example["answer"].strip(),
            steps=steps,
            final_answer=final_answer,
        )

    def extract_calculations(self) -> List[tuple[str, float, float]]:
        """Extract arithmetic calculations from the solution steps.

        Returns:
            List of tuples containing (expression, expected_result, actual_result)
        """
        calculations = []
        pattern = r"<<(.+?)=(.+?)>>"

        for step in self.steps:
            matches = re.finditer(pattern, step)
            for match in matches:
                expr, result = match.groups()
                try:
                    # Clean the expression and make it Python-safe
                    expr = expr.strip().replace("×", "*").replace("÷", "/")
                    actual = eval(
                        expr
                    )  # Note: eval is safe here as we control the input
                    expected = float(result)
                    calculations.append((expr, expected, actual))
                except:
                    continue

        return calculations


class GSM8KBenchmark(BaseBenchmark):

    name = "gsm8k"

    def __init__(self, seed: int | None = None, subset_size: int | None = None):
        super().__init__(seed, subset_size)

        # Validate inputs
        if subset_size is not None and subset_size <= 0:
            raise ValueError("subset_size must be positive")

        dataset = load_dataset("openai/gsm8k", "main")
        # self.train_data = [GSM8KExample.from_raw(ex) for ex in dataset["train"]]
        self.test_data = [GSM8KExample.from_raw(ex) for ex in dataset["test"]]

        self._data = [
            Problem(problem_id=str(i), statement=p.question, answer=p.final_answer, answer_discussion="\n".join(p.steps))
            for i, p in enumerate(self.test_data)
        ]

        # Create randomized subset if requested
        if subset_size is not None:
            random.seed(seed)
            self._data = random.sample(self._data, subset_size)

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
        try:
            answer_path = Path(agent_answer_dir) / "answer.txt"
            llm_answer = answer_path.read_text().strip()

            float_answer = float(llm_answer.strip().replace(",", "").replace(" ", ""))
            if abs(problem.answer - float_answer) < 1e-7:
                return 1.0, None, problem.answer_discussion
            else:
                return 0.0, None, problem.answer_discussion
        except Exception as e:
            logger.debug(f"Error in gsm8k scoring: {e}")
            return 0.0, str(e), problem.answer_discussion
