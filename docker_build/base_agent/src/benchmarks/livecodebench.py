# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""LiveCodeBench benchmark implementation."""

import zlib
import json
import pickle
import random
import base64
import logging
import subprocess

from enum import Enum
from typing import Sequence
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from dataclasses import dataclass

from .base import BaseBenchmark, Problem

logger = logging.getLogger(__name__)

# Import string used by LiveCodeBench
IMPORT_STRING = """
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string
import re
import datetime
import collections
import heapq
import bisect
import copy
import math
import random
import statistics
import itertools
import functools
import operator
import io
import sys
import json
sys.setrecursionlimit(50000)
"""


class Platform(Enum):
    """Platform enumeration for LiveCodeBench problems."""

    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    """Difficulty levels for LiveCodeBench problems."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TestType(Enum):
    """Test types for LiveCodeBench problems."""

    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    """A single test case for a LiveCodeBench problem."""

    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        """Convert types and validate test case data."""
        # Convert testtype to enum
        if isinstance(self.testtype, str):
            self.testtype = TestType(self.testtype)
        elif not isinstance(self.testtype, TestType):
            raise ValueError(f"Invalid testtype: {self.testtype}")

        # Handle input conversion
        if isinstance(self.input, (list, dict)):
            # Convert JSON-serializable objects to string
            self.input = json.dumps(self.input)
        elif not isinstance(self.input, str):
            # Convert simple types to string
            self.input = str(self.input)

        # Handle output conversion
        if isinstance(self.output, (list, dict)):
            # Convert JSON-serializable objects to string
            self.output = json.dumps(self.output)
        elif not isinstance(self.output, str):
            # Convert simple types to string
            self.output = str(self.output)

    def format_example(self) -> str:
        """Format this test case as a readable example."""
        try:
            if self.testtype == TestType.STDIN:
                return (
                    "Sample Input:\n"
                    f"{self.input}\n\n"
                    "Sample Output:\n"
                    f"{self.output}"
                )
            else:
                # For functional tests, try to parse as JSON
                try:
                    args = (
                        json.loads(self.input)
                        if isinstance(self.input, str)
                        else self.input
                    )
                    if not isinstance(args, (list, tuple)):
                        args = [args]  # Wrap single arguments
                except json.JSONDecodeError:
                    if isinstance(self.input, str) and "[" not in self.input:
                        # Single argument case
                        args = [self.input]
                    else:
                        args = [self.input]

                try:
                    expected = (
                        json.loads(self.output)
                        if isinstance(self.output, str)
                        else self.output
                    )
                except json.JSONDecodeError:
                    expected = self.output

                args_str = ", ".join(str(arg) for arg in args)
                return (
                    "Sample Call:\n"
                    f"solution({args_str})\n\n"
                    "Expected Output:\n"
                    f"{expected}"
                )
        except Exception as e:
            logger.warning(f"Error formatting example: {e}")
            return (
                "Sample Input:\n"
                f"{self.input}\n\n"
                "Sample Output:\n"
                f"{self.output}"
            )

    @staticmethod
    def create_test_case(data: dict) -> "Test | None":
        """Create a Test object from raw data with validation.

        Args:
            data: Dictionary containing test case data

        Returns:
            Test object or None if creation fails
        """
        try:
            # Validate required fields
            if not isinstance(data, dict):
                raise ValueError(f"Test case must be dict, got {type(data)}")

            required = {"input", "output", "testtype"}
            missing = required - set(data.keys())
            if missing:
                raise ValueError(f"Missing required fields: {missing}")

            # Create test object with type conversion
            return Test(
                input=data["input"], output=data["output"], testtype=data["testtype"]
            )
        except Exception as e:
            logger.warning(f"Failed to create test case: {e}")
            return None


@dataclass
class LiveCodeBenchProblem(Problem):
    """Problem subclass specifically for LiveCodeBench tasks."""

    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    @staticmethod
    def parse_test_cases(
        test_data: str | list, problem_id: str = "unknown"
    ) -> list[Test]:
        """Parse and validate test cases from raw data.

        Args:
            test_data: Test case data in various formats
            problem_id: Problem ID for error reporting

        Returns:
            List of validated Test objects

        Raises:
            ValueError: If parsing fails
        """
        try:
            # Handle direct list input
            if isinstance(test_data, list):
                result = []
                for case in test_data:
                    if test := Test.create_test_case(case):
                        result.append(test)
                if not result:
                    raise ValueError("No valid test cases in list")
                return result

            # Try decompressing private test cases
            if isinstance(test_data, str) and "=" in test_data:
                try:
                    compressed = test_data.encode("utf-8")
                    decompressed = zlib.decompress(base64.b64decode(compressed))
                    pickle_data = pickle.loads(decompressed)

                    if isinstance(pickle_data, list):
                        return LiveCodeBenchProblem.parse_test_cases(
                            pickle_data, problem_id
                        )
                    else:
                        test_cases = json.loads(pickle_data)
                        return LiveCodeBenchProblem.parse_test_cases(
                            test_cases, problem_id
                        )

                except Exception as e:
                    logger.warning(f"Problem {problem_id}: Failed to decompress: {e}")

            # Parse test cases based on format detection
            test_data = test_data.strip()
            first_line = test_data.split("\n", 1)[0].strip()

            if first_line.startswith("{") or first_line.startswith("["):
                # Functional test case format
                test_cases = json.loads(test_data)
                if isinstance(test_cases, dict):
                    test_cases = [test_cases]

                result = []
                for case in test_cases:
                    if test := Test.create_test_case(case):
                        result.append(test)

            else:
                # STDIN test case format
                input_lines = [
                    line.strip() for line in test_data.split("\n") if line.strip()
                ]
                test_cases = []
                for i in range(0, len(input_lines), 2):
                    if i + 1 < len(input_lines):
                        test_case = {
                            "input": input_lines[i] + "\n",
                            "output": input_lines[i + 1] + "\n",
                            "testtype": "stdin",
                        }
                        if test := Test.create_test_case(test_case):
                            test_cases.append(test)

                if not test_cases:
                    raise ValueError("No valid STDIN test cases parsed")
                return test_cases

            if not result:
                raise ValueError("No valid test cases parsed")

            return result

        except Exception as e:
            raise ValueError(
                f"Failed to parse test cases for problem {problem_id}: {e}"
            )

    @classmethod
    def from_raw(cls, example: dict) -> "LiveCodeBenchProblem":
        """Create a LiveCodeBenchProblem from a raw dataset example."""
        try:
            # Parse basic fields first
            platform = Platform(example["platform"])
            difficulty = Difficulty(example["difficulty"])
            contest_date = datetime.fromisoformat(example["contest_date"])
            metadata = json.loads(example["metadata"])

            # Parse public test cases - these must succeed
            try:
                public_tests = cls.parse_test_cases(
                    example["public_test_cases"], example["question_id"]
                )
                if not public_tests:
                    raise ValueError("No valid public test cases")
            except Exception as e:
                logger.debug(
                    f"Failed to parse public tests for {example['question_id']}: {e}"
                )
                raise ValueError(f"Could not load required public test cases: {e}")

            # Parse private test cases - these can fail gracefully
            try:
                private_tests = cls.parse_test_cases(
                    example["private_test_cases"], example["question_id"]
                )
                if not private_tests:
                    logger.warning(
                        f"No valid private tests for {example['question_id']}"
                    )
            except Exception as e:
                # logger.warning(f"Failed to parse private tests for {example['question_id']}: {e}")
                private_tests = []

            # Get test type from public tests
            test_type = public_tests[0].testtype

            # Generate appropriate starter code
            if test_type == TestType.STDIN:
                # TODO: fix STDIN tests
                raise NotImplementedError(f"We're skipping STDIN problems for now")

                starter_code = (
                    "# Problem requires reading from stdin and writing to stdout\n"
                    "# Input format will be exactly as shown in the examples\n"
                    "# Remember to handle all input processing and print output\n\n"
                )
            else:  # TestType.FUNCTIONAL
                starter_code = (
                    "def solution(*args):\n"
                    "    # Implement your solution here\n"
                    "    # Function will receive exactly the arguments from the example\n"
                    "    # Return value must match the example output format\n"
                    "    pass\n"
                )

            # Format problem statement sections
            problem_info = (
                f"# {example['question_title']}\n\n"
                # f"## Problem ID: {example['question_id']}\n"
                # f"## Platform: {platform.value}\n"
                # f"## Contest: {example['contest_id']}\n"
                # f"## Date: {contest_date}\n"
                f"## Difficulty: {difficulty.value}\n\n"
                f"## Problem Description:\n\n"
                f"{example['question_content']}\n\n"
            )

            submission_instructions = (
                "## Submission Instructions:\n\n"
                "Please write your solution in a file named 'solution.py' "
                "in your work directory.\n\n"
            )

            if test_type == TestType.FUNCTIONAL:
                submission_instructions += (
                    "Your solution should implement the following function:\n"
                    "```python\n"
                    f"{starter_code}"
                    "```\n"
                    "The function will be called with the specified arguments and "
                    "your return value will be compared with the expected output.\n\n"
                )
            else:
                submission_instructions += (
                    "Your solution should read input from standard input (stdin) and "
                    "write output to standard output (stdout).\n"
                    "Input format and output format are shown in the examples below.\n\n"
                    "Use the following Python methods to read input:\n"
                    "- input() to read a line\n"
                    "- sys.stdin.read() to read all input\n"
                    "- sys.stdin.readline() to read a line\n"
                    "- sys.stdin.readlines() to read all lines\n\n"
                    "Write your output using print() statements.\n\n"
                )

            test_cases = "## Example Test Cases:\n\n"
            for i, test in enumerate(public_tests, 1):
                test_cases += f"### Example {i}:\n{test.format_example()}\n\n"

            full_statement = problem_info + submission_instructions + test_cases

            # Create and return problem instance
            return cls(
                problem_id=example["question_id"],
                statement=full_statement,
                answer=None,
                answer_discussion=None,
                question_title=example["question_title"],
                question_content=example["question_content"],
                platform=platform,
                question_id=example["question_id"],
                contest_id=example["contest_id"],
                contest_date=contest_date,
                starter_code=starter_code,
                difficulty=difficulty,
                public_test_cases=public_tests,
                private_test_cases=private_tests,
                metadata=metadata,
            )

        except Exception as e:
            raise ValueError(
                f"Failed to create problem {example.get('question_id', 'unknown')}: {e}"
            )

    def create_test(self, i: int, test: Test) -> str:
        """Create a test file for a specific test case."""
        if test.testtype == TestType.STDIN:
            template = f"""
import io
import sys
from decimal import Decimal
from unittest.mock import patch, mock_open

def run_test():
    inputs = {repr(test.input)}
    expected_output = {repr(test.output)}

    # Create list of input lines without stripping newlines
    input_lines = []
    remaining = inputs
    while remaining:
        if '\\n' in remaining:
            line, remaining = remaining.split('\\n', 1)
            input_lines.append(line)
        else:
            if remaining:
                input_lines.append(remaining)
            remaining = ''

    # Create input line generator
    def input_generator():
        yield from input_lines
        while True:  # Cycle last input if more needed
            yield input_lines[-1] if input_lines else ''

    inputs_line_iterator = input_generator()

    @patch('builtins.input', side_effect=lambda: next(inputs_line_iterator))
    @patch('builtins.open', mock_open(read_data=inputs))
    @patch('sys.stdin', io.StringIO(inputs))
    @patch('sys.stdin.readline', lambda *args: next(inputs_line_iterator) + '\\n')
    @patch('sys.stdin.readlines', lambda *args: [line + '\\n' for line in input_lines])
    @patch('sys.stdin.read', lambda *args: inputs)
    def run_with_input(_input):
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Add imports and execute solution
            with open('solution.py') as f:
                exec(f.read(), globals())

            # Get output with consistent line endings
            output = sys.stdout.getvalue().rstrip()
            expected = expected_output.rstrip()

            output_lines = output.split('\\n')
            expected_lines = expected.split('\\n')

            if len(output_lines) != len(expected_lines):
                print(f"FAILED: Wrong number of lines")
                print(f"Expected {{len(expected_lines)}} lines, got {{len(output_lines)}}")
                print(f"Expected output: {{expected}}")
                print(f"Got output: {{output}}")
                return

            for j, (out_line, exp_line) in enumerate(zip(output_lines, expected_lines)):
                out_line = out_line.rstrip()
                exp_line = exp_line.rstrip()

                if out_line == exp_line:
                    continue

                # Try numeric comparison for float values
                try:
                    # First try decimal for exact comparison
                    out_nums = [Decimal(x) for x in out_line.split()]
                    exp_nums = [Decimal(x) for x in exp_line.split()]
                    if len(out_nums) != len(exp_nums):
                        raise ValueError()
                    if out_nums == exp_nums:
                        continue

                    # Fall back to float for approximate comparison
                    out_nums = [float(x) for x in out_line.split()]
                    exp_nums = [float(x) for x in exp_line.split()]
                    if all(abs(o - e) < 1e-6 for o, e in zip(out_nums, exp_nums)):
                        continue
                except (ValueError, TypeError, decimal.InvalidOperation):
                    pass

                print(f"FAILED: Wrong answer on line {{j+1}}")
                print(f"Expected: {{exp_line}}")
                print(f"Got: {{out_line}}")
                return

            print("PASSED")

        except Exception as e:
            print(f"ERROR: {{str(e)}}")
            import traceback
            traceback.print_exc()
        finally:
            sys.stdout = old_stdout

    run_with_input(None)

if __name__ == '__main__':
    run_test()
"""
        else:  # TestType.FUNCTIONAL
            # args = json.loads({repr(test.input)})
            template = f"""
import json
from decimal import Decimal
from solution import solution

def run_test():
    try:
        # Parse input arguments

        lines = {repr(test.input)}.strip().splitlines()
        args = [json.loads(line) for line in lines]

        expected = json.loads({repr(test.output)})

        if not isinstance(args, list):
            args = [args]

        # Call function and convert tuples to lists
        result = solution(*args)
        if isinstance(result, tuple):
            result = list(result)
        if isinstance(expected, tuple):
            expected = list(expected)

        # Exact comparison first
        if result == expected:
            print("PASSED")
            return

        # Try numeric comparison for float values
        try:
            if isinstance(result, (list, tuple)) and isinstance(expected, (list, tuple)):
                if len(result) != len(expected):
                    raise ValueError("Different length sequences")
                if all(isinstance(r, (int, float)) for r in result) and all(isinstance(e, (int, float)) for e in expected):
                    if all(abs(float(r) - float(e)) < 1e-6 for r, e in zip(result, expected)):
                        print("PASSED")
                        return
            elif isinstance(result, (int, float)) and isinstance(expected, (int, float)):
                if abs(float(result) - float(expected)) < 1e-6:
                    print("PASSED")
                    return
        except (TypeError, ValueError):
            pass

        print(f"FAILED: Expected {{expected}}, got {{result}}")

    except Exception as e:
        print(f"ERROR: {{str(e)}}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_test()
"""
        return template

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """Score the submitted answer for the LiveCodeBench problem."""
        try:
            if not isinstance(problem, LiveCodeBenchProblem):
                return 0.0, "Invalid problem type", None

            # Check for solution file
            solution_file = Path(agent_workdir) / "solution.py"
            if not solution_file.exists():
                return 0.0, "No solution.py file found in work directory", None

            # Read the solution code
            solution_code = solution_file.read_text().strip()
            if not solution_code:
                return 0.0, "Empty solution.py file", None

            # Add imports if not present
            if IMPORT_STRING not in solution_code:
                solution_code = IMPORT_STRING + "\n\n" + solution_code
                solution_file.write_text(solution_code)

            test_cases = problem.public_test_cases + problem.private_test_cases
            results = []
            error_details = []

            # Create test files
            workdir_path = Path(agent_workdir)

            for i, test in enumerate(test_cases, 1):
                test_file = workdir_path / f"test_{i}.py"
                test_content = self.create_test(i, test)
                test_file.write_text(test_content)

                # Run test with timeout in container
                result = subprocess.run(
                    [
                        "docker",
                        "exec",
                        container_name,
                        "timeout",
                        "20",  # Add timeout
                        "python",
                        f"/home/agent/workdir/test_{i}.py",
                    ],
                    capture_output=True,
                    text=True,
                    env={"PYTHONPATH": "/home/agent/workdir"},
                )

                # Process result
                output = result.stdout.strip()
                error = result.stderr.strip()

                if "PASSED" in output:
                    results.append(True)
                else:
                    results.append(False)
                    error_msg = f"Test {i} failed:"
                    if error:
                        error_msg += f"\nError: {error}"
                    if output:
                        error_msg += f"\nOutput: {output}"
                    error_details.append(error_msg)

            # Calculate final score
            if not results:
                return 0.0, "No test results", None

            score = sum(1 for r in results if r) / len(results)
            if score < 1.0 and error_details:
                return score, "\n".join(error_details), None

            return score, None, None

        except Exception as e:
            logger.error(f"Error scoring LiveCodeBench problem: {e}")
            return 0.0, str(e), None


class LiveCodeBenchmark(BaseBenchmark):
    """Benchmark for evaluating code generation using the LiveCodeBench dataset."""

    name = "livecodebench"

    def __init__(self, seed: int = 2, subset_size: int | None = 8):
        """Initialize the LiveCodeBench benchmark.

        Args:
            seed: Random seed for shuffling problems
            subset_size: Number of problems to use (or None for all)
        """
        super().__init__()
        split: str = "test"
        self.split = split
        self.seed = seed
        self.subset_size = subset_size

        try:
            dataset = load_dataset(
                "livecodebench/code_generation_lite",
                split="test",
                trust_remote_code=True,
                version_tag="release_v5",
            )
            dataset = load_dataset(
                "livecodebench/code_generation_lite",
                split="test",
                trust_remote_code=True,
                version_tag="release_v5",
            )
            all_examples = list(dataset)

            random.seed(seed)
            random.shuffle(all_examples)

            # if subset_size is not None:
            #     all_examples = all_examples[: min(subset_size, len(all_examples))]

            self._all_problems = []
            for ex in all_examples:
                try:
                    new_problem = LiveCodeBenchProblem.from_raw(ex)
                    self._all_problems.append(new_problem)

                    if (
                        subset_size is not None
                        and len(self._all_problems) >= subset_size
                    ):
                        break
                except Exception as e:
                    logger.debug(f"Skipping: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error loading LiveCodeBench dataset: {e}")
            self._all_problems = [
                Problem(
                    problem_id="error",
                    statement="Error loading mock LiveCodeBench dataset",
                    answer=str(e),
                    answer_discussion=None,
                )
            ]

    @property
    def problems(self) -> Sequence[Problem]:
        return self._all_problems if self._all_problems is not None else []

    async def setup_problem(
        self,
        problem: Problem,
        problem_data_dir: Path,
        container_name: str,
    ) -> None:
        """Optional setup for this problem."""
        try:
            if not isinstance(problem, LiveCodeBenchProblem):
                return

            # Create starter solution.py file with imports
            solution_file = problem_data_dir / "solution.py"
            solution_file.write_text(f"{IMPORT_STRING}\n{problem.starter_code}")
        except Exception as e:
            logger.error(f"Error setting up LiveCodeBench problem: {e}")

    def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """Score a LiveCodeBench problem solution."""
        if not isinstance(problem, LiveCodeBenchProblem):
            return 0.0, "Not a LiveCodeBench problem", None
        return problem.score_problem(
            problem, agent_workdir, agent_answer_dir, container_name
        )


# Test entrypoint
if __name__ == "__main__":
    import asyncio

    async def main():
        benchmark = LiveCodeBenchmark(subset_size=20)
        problems = benchmark.problems
        print(f"Loaded {len(problems)} problems from LiveCodeBench dataset")

        if problems:
            print("\nExample problem:")
            example = problems[0]
            print(f"Problem ID: {example.problem_id}")
            print(f"Statement:\n{example.statement}")

            if isinstance(example, LiveCodeBenchProblem):
                print(f"\nPlatform: {example.platform}")
                print(f"Difficulty: {example.difficulty}")
                print("\nPublic test cases:")
                for test in example.public_test_cases:
                    print(f"  Input: {test.input}")
                    print(f"  Output: {test.output}")
                    print(f"  Type: {test.testtype}\n")

    asyncio.run(main())
