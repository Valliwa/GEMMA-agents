# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import string
import logging

from typing import List, Tuple
from pathlib import Path
from datasets import load_dataset

from .base import BaseBenchmark, Problem

logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize answer text for consistent comparison.

    This performs case normalization, whitespace standardization,
    and removes articles and punctuation for robust matching.

    Args:
        text: The text string to normalize

    Returns:
        Normalized text string
    """
    # Convert to lowercase and strip whitespace
    text = text.lower().strip()

    # Remove punctuation except in numbers (e.g. '3.14')
    def remove_punct(token: str) -> str:
        if not any(c.isdigit() for c in token):
            return "".join(c for c in token if c not in string.punctuation)
        return token

    # Remove articles and normalize whitespace
    tokens = text.split()
    tokens = [remove_punct(token) for token in tokens]
    tokens = [token for token in tokens if token not in ["a", "an", "the"]]
    return " ".join(tokens)


def calc_f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score between prediction and ground truth.

    The F1 score is computed based on token overlap after normalization.
    Both strings are normalized and tokenized before computing precision
    and recall scores.

    Args:
        prediction: The predicted answer string
        ground_truth: The ground truth answer string

    Returns:
        F1 score between 0.0 and 1.0
    """
    # Normalize and tokenize
    pred_tokens = set(normalize_text(prediction).split())
    truth_tokens = set(normalize_text(ground_truth).split())

    # Handle empty cases
    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0

    # Calculate overlap
    common_tokens = pred_tokens & truth_tokens

    # Handle no overlap case
    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)

    # Calculate F1
    return (2 * precision * recall) / (precision + recall)


def evaluate_prediction(prediction: str, answers: List[str]) -> Tuple[float, float]:
    """Evaluate a prediction against multiple valid answers.

    Calculates both exact match and F1 scores, taking the maximum
    score across all valid answer options.

    Args:
        prediction: The predicted answer string
        answers: List of valid answer strings

    Returns:
        Tuple of (max_exact_match, max_f1) scores
    """
    if not answers:
        return 0.0, 0.0

    exact_matches = []
    f1_scores = []

    prediction = prediction.strip()

    for answer in answers:
        if not answer.strip():
            continue

        # Calculate exact match
        exact_matches.append(
            1.0 if normalize_text(prediction) == normalize_text(answer) else 0.0
        )

        # Calculate F1
        f1_scores.append(calc_f1_score(prediction, answer))

    if not exact_matches:
        return 0.0, 0.0

    return max(exact_matches), max(f1_scores)


class DROPBenchmark(BaseBenchmark):
    """DROP (Discrete Reasoning Over Paragraphs) benchmark implementation.

    This benchmark evaluates reading comprehension requiring discrete reasoning
    over paragraphs. Problems involve reading a passage and answering questions
    that may require numerical computation, list manipulation, or other forms
    of reasoning beyond simple text comprehension.

    The benchmark uses both F1 and Exact Match (EM) scores for evaluation.
    Each question can have multiple valid answers, and the scoring system
    accounts for this by taking the maximum score across all valid answers.
    """

    name = "drop"

    def __init__(self, seed: int | None, subset_size: int | None = 20) -> None:
        """Initialize the DROP benchmark.

        Loads the DROP dataset from HuggingFace and converts examples
        into Problem instances.

        Args:
            seed: Random seed for consistent answer shuffling across runs
            subset_size: Number of questions to sample (None for full dataset)
        """
        super().__init__(seed, subset_size)

        # Validate inputs
        if subset_size is not None and subset_size <= 0:
            raise ValueError("subset_size must be positive")

        try:
            # Load DROP dataset from HuggingFace
            dataset = load_dataset("ucinlp/drop", split="validation")

            # Process examples into problems
            self._data = []
            for i, example in enumerate(dataset):
                # Extract answer spans
                spans = example["answers_spans"]["spans"]
                if not spans:
                    continue

                # Combine passage and question
                problem_text = (
                    f"You will be asked to read a passage and answer a question.\n\n"
                    f"Passage: {example['passage']}\n"
                    f"Question: {example['question']}"
                )

                self._data.append(
                    Problem(
                        problem_id=str(i),
                        statement=problem_text,
                        answer=spans,  # List of valid answer strings
                        answer_discussion=None,
                    )
                )

            # Create randomized subset if requested
            if subset_size is not None:
                random.seed(seed)
                self._data = random.sample(self._data, subset_size)

        except Exception as e:
            logger.error(f"Error loading DROP dataset: {e}")
            # Provide empty dataset with error message
            self._data = [
                Problem(
                    problem_id="error",
                    statement="Error loading DROP dataset",
                    answer=[],
                    answer_discussion=None
                )
            ]

    @property
    def problems(self) -> list[Problem]:
        """Return list of DROP problems.

        Returns:
            list[Problem]: List of Problem instances representing the DROP dataset.
        """
        return self._data

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """Score an LLM's answer against the ground truth.

        Uses both Exact Match (EM) and F1 scores to evaluate the answer.
        For this implementation, we use the F1 score as it provides a more
        granular measure of partial correctness.

        Args:
            problem: Problem instance containing the reference answer
            agent_workdir: The agent's working directory where any files would have been written
            agent_answer_dir: Where the agent should have written its answer.txt
            container_name: The name of the running docker container where the agent run

        Returns:
            float: F1 score between 0.0 and 1.0. A score of 1.0 indicates a perfect match.
        """
        try:
            answer_path = Path(agent_answer_dir) / "answer.txt"
            llm_answer = answer_path.read_text().strip()

            # Evaluate prediction against all valid answers
            _, f1_score = evaluate_prediction(llm_answer, problem.answer)
            return f1_score, None, None
        except Exception as e:
            logger.error(f"Error scoring DROP answer: {e}")
            return 0.0, str(e), None

    def get_problem(self, problem_id: str) -> Problem | None:
        """Get a specific problem by ID.

        Provides direct lookup using the list index since problem_ids
        are sequential integers converted to strings.

        Args:
            problem_id (str): The ID of the problem to retrieve

        Returns:
            Problem | None: The requested Problem instance or None if not found
        """
        try:
            idx = int(problem_id)
            if 0 <= idx < len(self._data):
                return self._data[idx]
        except ValueError:
            pass
        return None
