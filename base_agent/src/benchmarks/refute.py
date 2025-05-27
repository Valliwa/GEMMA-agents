# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""REFUTE Benchmark Implementation.

This benchmark tests an agent's ability to create counterexamples for incorrect
algorithmic solutions. Based on the REFUTE dataset from bethgelab.
"""

import re
import json
import random
import logging
import subprocess
import tempfile
import shutil

from typing import List, Dict, Any, Tuple, ClassVar, Optional
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from dataclasses import dataclass, field

from .base import BaseBenchmark, Problem

logger = logging.getLogger(__name__)


@dataclass
class RefuteExample:
    """A single REFUTE example containing a programming problem and an incorrect solution."""

    problem_id: str
    problem_statement: str
    problem_rating: int
    author_rating: int
    input_format: str
    output_format: str
    tags: List[str]
    incorrect_code: str
    incorrect_code_lang: str
    correct_cpp: str
    correct_python: str
    validator: str
    example_input: str
    example_output: str
    note: str = ""
    time_limit: float = 3.0
    memory_limit: int = 256
    fail_tc: Optional[int] = None

    @classmethod
    def from_raw(cls, example: dict) -> "RefuteExample":
        """Create a RefuteExample from a raw dataset example."""
        # Parse tags from string representation
        tags_str = example.get("tags", "[]")
        try:
            tags = json.loads(tags_str.replace("'", '"'))
        except:
            tags = re.findall(r"'([^']*)'", tags_str)

        return cls(
            problem_id=example["problem_id"],
            problem_statement=example["statement"],
            problem_rating=example["problem_rating"],
            author_rating=example["author_rating"],
            input_format=example["input"],
            output_format=example["output"],
            tags=tags,
            incorrect_code=example["wrong_code"],
            incorrect_code_lang=example["wrong_code_lang"],
            correct_cpp=example["correct_cpp"],
            correct_python=example.get("correct_python", ""),
            validator=example["validator"],
            example_input=example["example_input"],
            example_output=example["example_output"],
            note=example.get("note", ""),
            time_limit=float(example.get("time_limit", 3.0)) / 1000,  # Convert to seconds
            memory_limit=int(example.get("memory_limit", 256)),
            fail_tc=example.get("fail_tc")
        )

    def format_problem_statement(self) -> str:
        """Format the complete problem statement with proper formatting."""
        # Helper function to normalize LaTeX formatting
        def normalize_latex(text: str) -> str:
            # Replace triple dollar signs with single dollars for inline math
            text = re.sub(r'\${3}(.*?)\${3}', r'$\1$', text)
            # Clean up other potential LaTeX formatting issues
            text = re.sub(r'\\ldots', '...', text)
            return text

        # Clean and normalize all text content
        statement = normalize_latex(self.problem_statement)
        input_format = normalize_latex(self.input_format)
        output_format = normalize_latex(self.output_format)
        example_input = self.example_input.strip()
        example_output = self.example_output.strip()

        # Build the problem statement with clean formatting
        full_statement = f"""# REFUTE Challenge: Create a Counterexample

## Problem ID: {self.problem_id}
## Problem Rating: {self.problem_rating}
## Tags: {', '.join(self.tags)}

## Problem Statement:
{statement}

## Input Format:
{input_format}

## Output Format:
{output_format}

## Example Input:
```
{example_input}
```

## Example Output:
```
{example_output}
```
"""
        if self.note:
            full_statement += f"## Note:\n{normalize_latex(self.note)}\n\n"

        # Format the incorrect solution with appropriate language tagging
        lang = "cpp" if "C++" in self.incorrect_code_lang else "python"
        full_statement += f"""## Incorrect Solution ({self.incorrect_code_lang}):
```{lang}
{self.incorrect_code}
```

## Your Task:
Create a counterexample - a valid input that makes the incorrect solution produce a wrong answer.
Your counterexample must follow all input constraints specified in the problem.

## How Your Solution Will Be Evaluated:
1. Your counterexample will be validated against the input constraints
2. Your counterexample will be run through both the incorrect solution shown above and a correct solution
3. If the two solutions produce different outputs for your input, your counterexample is valid
4. Your score will be 1.0 for a valid counterexample and 0.0 otherwise

Submit your counterexample using the `submit_answer` tool. Make sure your answer:
1. Follows the exact input format specified in the problem
2. Satisfies all constraints mentioned in the problem
3. Causes the incorrect solution to produce a wrong output

For example, if the problem asks for an integer n followed by n elements,
and you believe the solution fails for n=3 with elements [1, 2, 3], your answer should be:
```
3
1 2 3
```
"""
        return full_statement


class RefuteBenchmark(BaseBenchmark):
    """Benchmark for evaluating an agent's ability to create counterexamples for incorrect solutions."""

    name: ClassVar[str] = "refute"

    def __init__(self, seed: int | None = None, subset_size: int | None = None):
        """Initialize the REFUTE benchmark.

        Args:
            seed: Random seed for shuffling problems
            subset_size: Number of problems to use (or None for all)
        """
        super().__init__(seed, subset_size)

        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("bethgelab/REFUTE", token="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")
            all_examples = list(dataset["train"])

            # Shuffle and sample the examples
            if seed is not None:
                random.seed(seed)
                random.shuffle(all_examples)

            # Create RefuteExample objects
            parsed_examples = []
            for ex in all_examples:
                try:
                    parsed_examples.append(RefuteExample.from_raw(ex))
                except Exception as e:
                    logger.warning(f"Failed to parse example {ex.get('problem_id', 'unknown')}: {e}")
                    continue

            # Create subset if requested
            if subset_size is not None:
                parsed_examples = parsed_examples[:min(subset_size, len(parsed_examples))]

            # Convert to Problem objects required by the base benchmark
            self._problems = [
                Problem(
                    problem_id=ex.problem_id,
                    statement=ex.format_problem_statement(),
                    answer=ex,  # Store the full example as the answer for scoring
                    answer_discussion=None
                )
                for ex in parsed_examples
            ]

            logger.info(f"Loaded {len(self._problems)} problems from REFUTE dataset")

        except Exception as e:
            logger.error(f"Error loading REFUTE dataset: {e}")
            self._problems = []

    @property
    def problems(self) -> List[Problem]:
        """Get the list of problems for this benchmark."""
        return self._problems

    async def setup_problem(
        self,
        problem: Problem,
        problem_data_dir: Path,
        container_name: str,
    ) -> None:
        """Set up the problem environment before execution.

        This creates necessary files in the container for testing the counterexample.
        """
        try:
            if not isinstance(problem.answer, RefuteExample):
                return

            ex = problem.answer

            # Create directory for validator and solution files
            validator_dir = problem_data_dir / "validator"
            validator_dir.mkdir(exist_ok=True)

            # Write the validator script
            validator_path = validator_dir / "validator.py"
            validator_path.write_text(ex.validator)

            # Set up the incorrect solution
            incorrect_solution_path = validator_dir / "incorrect_solution"
            if "C++" in ex.incorrect_code_lang:
                incorrect_solution_path = incorrect_solution_path.with_suffix(".cpp")
                incorrect_solution_path.write_text(ex.incorrect_code)

                # Compile the C++ solution in the container
                compile_cmd = [
                    "docker", "exec", container_name,
                    "g++", "-std=c++17", "-O2",
                    f"/home/agent/workdir/validator/incorrect_solution.cpp",
                    "-o", "/home/agent/workdir/validator/incorrect_solution"
                ]

                try:
                    subprocess.run(compile_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to compile incorrect solution: {e.stderr}")
                    # Create a dummy executable that indicates compilation error
                    error_script = (
                        "#!/bin/bash\n"
                        "echo 'Compilation Error in incorrect solution'\n"
                        "exit 1\n"
                    )
                    error_path = validator_dir / "incorrect_solution"
                    error_path.write_text(error_script)
                    error_path.chmod(0o755)
            else:
                # Handle Python solutions
                incorrect_solution_path = incorrect_solution_path.with_suffix(".py")
                incorrect_solution_path.write_text(ex.incorrect_code)

            # Set up the correct solution
            correct_solution_path = validator_dir / "correct_solution"
            if ex.correct_python:
                correct_solution_path = correct_solution_path.with_suffix(".py")
                correct_solution_path.write_text(ex.correct_python)
            else:
                correct_solution_path = correct_solution_path.with_suffix(".cpp")
                correct_solution_path.write_text(ex.correct_cpp)

                # Compile the C++ solution in the container
                compile_cmd = [
                    "docker", "exec", container_name,
                    "g++", "-std=c++17", "-O2",
                    f"/home/agent/workdir/validator/correct_solution.cpp",
                    "-o", "/home/agent/workdir/validator/correct_solution"
                ]

                try:
                    subprocess.run(compile_cmd, check=True, capture_output=True)
                except subprocess.CalledProcessError as e:
                    logger.warning(f"Failed to compile correct solution: {e.stderr}")
                    # Create a dummy executable that indicates compilation error
                    error_script = (
                        "#!/bin/bash\n"
                        "echo 'Compilation Error in correct solution'\n"
                        "exit 1\n"
                    )
                    error_path = validator_dir / "correct_solution"
                    error_path.write_text(error_script)
                    error_path.chmod(0o755)

            # Add a README to help the agent understand the task
            readme_path = problem_data_dir / "README.md"
            readme_path.write_text(
                """# REFUTE Challenge

Your task is to create a counterexample that makes the incorrect solution fail.

## Instructions:
1. The problem statement and incorrect solution are provided in the prompt.
2. Analyze the code to identify potential bugs or edge cases.
3. Create a counterexample that follows the input format and constraints.
4. Submit your counterexample using the `submit_answer` tool.
5. Your counterexample should cause the incorrect solution to produce a wrong output.

Good luck!
"""
            )

            # Add a verification script that the agent can use to test counterexamples
            verify_path = problem_data_dir / "verify_counterexample.py"
            verify_path.write_text(
                """#!/usr/bin/env python3
import os
import sys
import subprocess
import tempfile

def main():
    # Ask the user for input if not provided
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            counterexample = f.read()
    else:
        print("Enter your counterexample input (press Ctrl+D when finished):")
        counterexample = sys.stdin.read()

    if not counterexample.strip():
        print("Error: Empty counterexample!")
        return

    # Create temporary files for output
    incorrect_output_path = tempfile.mktemp()
    correct_output_path = tempfile.mktemp()

    try:
        # Validate input
        print("Validating input...")
        try:
            validator_result = subprocess.run(
                ['python', 'validator/validator.py'],
                input=counterexample,
                text=True,
                capture_output=True,
                timeout=10
            )
            if validator_result.returncode != 0:
                print("Error: Input validation failed!")
                print(validator_result.stderr)
                return
            print("Input validation successful!")
        except subprocess.CalledProcessError as e:
            print("Error: Input validation failed!")
            print(e)
            return
        except subprocess.TimeoutExpired:
            print("Error: Input validation timed out - input might be too complex.")
            return

        # Run the incorrect solution
        print("Running incorrect solution...")
        if os.path.exists('validator/incorrect_solution.py'):
            incorrect_cmd = ['python', 'validator/incorrect_solution.py']
        else:
            incorrect_cmd = ['validator/incorrect_solution']

        try:
            with open(incorrect_output_path, 'w') as output_file:
                subprocess.run(
                    incorrect_cmd,
                    input=counterexample,
                    text=True,
                    stdout=output_file,
                    stderr=subprocess.PIPE,
                    timeout=30
                )
        except subprocess.TimeoutExpired:
            print("Error: Incorrect solution timed out. Try a simpler counterexample.")
            return
        except subprocess.CalledProcessError as e:
            print(f"Error: Incorrect solution crashed. This might be a valid counterexample.")
            print(f"Error message: {e}")
            with open(incorrect_output_path, 'w') as f:
                f.write(f"CRASH: {e}")

        # Run the correct solution
        print("Running correct solution...")
        if os.path.exists('validator/correct_solution.py'):
            correct_cmd = ['python', 'validator/correct_solution.py']
        else:
            correct_cmd = ['validator/correct_solution']

        try:
            with open(correct_output_path, 'w') as output_file:
                subprocess.run(
                    correct_cmd,
                    input=counterexample,
                    text=True,
                    stdout=output_file,
                    stderr=subprocess.PIPE,
                    timeout=30
                )
        except subprocess.TimeoutExpired:
            print("Error: Correct solution timed out. Try a simpler counterexample.")
            return
        except subprocess.CalledProcessError as e:
            print(f"Error: Correct solution crashed. This is unexpected.")
            print(f"Error message: {e}")
            with open(correct_output_path, 'w') as f:
                f.write(f"CRASH: {e}")

        # Read the outputs
        with open(incorrect_output_path, 'r') as f:
            incorrect_output = f.read().strip()

        with open(correct_output_path, 'r') as f:
            correct_output = f.read().strip()

        # Compare outputs
        if incorrect_output != correct_output:
            print("SUCCESS! Valid counterexample found!")
            print("\nYour input:")
            print("```")
            print(counterexample.strip())
            print("```")
            print("\nIncorrect solution output:")
            print("```")
            print(incorrect_output)
            print("```")
            print("\nCorrect solution output:")
            print("```")
            print(correct_output)
            print("```")
            print("\nYou can submit this counterexample using the submit_answer tool.")
        else:
            print("Both solutions produced the same output. Not a counterexample.")
            print("\nOutput:")
            print("```")
            print(correct_output)
            print("```")

    finally:
        # Clean up temp files
        for path in [incorrect_output_path, correct_output_path]:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except:
                    pass

if __name__ == "__main__":
    main()
"""
            )

            # Make the verification script executable
            verify_path.chmod(0o755)

        except Exception as e:
            logger.error(f"Error setting up REFUTE problem: {e}")

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> tuple[float, str | None, str | None]:
        """
        Score the submitted counterexample.

        Args:
            problem: The problem being scored
            agent_workdir: Path to the agent's working directory
            agent_answer_dir: Path to the agent's answer directory
            container_name: Name of the Docker container for execution

        Returns:
            A tuple of (score, error_message, discussion)
            score is 1.0 for valid counterexample, 0.0 otherwise
        """
        try:
            if not isinstance(problem.answer, RefuteExample):
                return 0.0, "Invalid problem type", None

            # Create paths
            workdir_path = Path(agent_workdir)
            answer_dir_path = Path(agent_answer_dir)
            validator_dir = workdir_path / "validator"

            # Check for the answer file
            answer_txt_path = answer_dir_path / "answer.txt"
            if not answer_txt_path.exists():
                return 0.0, "No answer.txt file found", None

            # Read the counterexample
            counterexample = answer_txt_path.read_text()
            if not counterexample.strip():
                return 0.0, "Empty counterexample", None

            ex = problem.answer

            # Create temporary output files
            incorrect_output_path = workdir_path / "incorrect_output.txt"
            correct_output_path = workdir_path / "correct_output.txt"

            # Compilation issues and other background information
            compilation_issues = []
            additional_info = []

            # First, validate the input using the validator
            try:
                validator_result = subprocess.run(
                    ["docker", "exec", container_name, "python",
                     "/home/agent/workdir/validator/validator.py"],
                    input=counterexample,
                    text=True,
                    capture_output=True,
                    timeout=30
                )
                if validator_result.returncode != 0:
                    return 0.0, f"Input validation failed: {validator_result.stderr}", None
            except Exception as e:
                return 0.0, f"Input validation error: {str(e)}", None

            # Execute the incorrect solution
            if "C++" in ex.incorrect_code_lang:
                incorrect_cmd = ["docker", "exec", container_name,
                                "/home/agent/workdir/validator/incorrect_solution"]
            else:
                incorrect_cmd = ["docker", "exec", container_name, "python",
                                "/home/agent/workdir/validator/incorrect_solution.py"]

            try:
                incorrect_result = subprocess.run(
                    incorrect_cmd,
                    input=counterexample,
                    text=True,
                    capture_output=True,
                    timeout=30
                )
                incorrect_output = incorrect_result.stdout.strip()
                if incorrect_result.returncode != 0:
                    incorrect_output = f"CRASH: {incorrect_result.stderr}"
                    additional_info.append(f"The incorrect solution crashed with: {incorrect_result.stderr}")
            except subprocess.TimeoutExpired:
                incorrect_output = "TIMEOUT"
                additional_info.append("The incorrect solution timed out.")
            except Exception as e:
                incorrect_output = f"ERROR: {str(e)}"
                additional_info.append(f"The incorrect solution encountered an error: {str(e)}")

            with open(incorrect_output_path, "w") as f:
                f.write(incorrect_output)

            # Execute the correct solution
            if ex.correct_python:
                correct_cmd = ["docker", "exec", container_name, "python",
                              "/home/agent/workdir/validator/correct_solution.py"]
            else:
                correct_cmd = ["docker", "exec", container_name,
                              "/home/agent/workdir/validator/correct_solution"]

            try:
                correct_result = subprocess.run(
                    correct_cmd,
                    input=counterexample,
                    text=True,
                    capture_output=True,
                    timeout=30
                )
                correct_output = correct_result.stdout.strip()
                if correct_result.returncode != 0:
                    additional_info.append(f"The correct solution failed with: {correct_result.stderr}")
                    return 0.0, None, f"Correct solution error: The solution that was supposed to be correct appears to have issues with the counterexample. {' '.join(additional_info)}"
            except subprocess.TimeoutExpired:
                additional_info.append("The correct solution timed out.")
                return 0.0, None, f"Correct solution timed out: The solution that was supposed to be correct took too long to process the counterexample. {' '.join(additional_info)}"
            except Exception as e:
                additional_info.append(f"The correct solution encountered an error: {str(e)}")
                return 0.0, None, f"Correct solution error: {str(e)}. {' '.join(additional_info)}"

            with open(correct_output_path, "w") as f:
                f.write(correct_output)

            # Compare outputs
            if incorrect_output != correct_output:
                # Check if incorrect output is a crash or timeout
                if incorrect_output.startswith(("CRASH:", "ERROR:", "TIMEOUT")):
                    discussion = (
                        f"Valid counterexample found! The incorrect solution failed to execute properly.\n\n"
                        f"Input:\n{counterexample}\n\n"
                        f"Incorrect solution output: {incorrect_output}\n\n"
                        f"Correct solution output:\n{correct_output}"
                    )
                    if additional_info:
                        discussion += f"\n\nAdditional information: {' '.join(additional_info)}"
                else:
                    discussion = (
                        f"Valid counterexample found!\n\n"
                        f"Input:\n{counterexample}\n\n"
                        f"Incorrect solution output:\n{incorrect_output}\n\n"
                        f"Correct solution output:\n{correct_output}"
                    )
                    if additional_info:
                        discussion += f"\n\nAdditional information: {' '.join(additional_info)}"
                return 1.0, None, discussion
            else:
                discussion = (
                    f"Both solutions produced the same output:\n{correct_output}\n\n"
                    f"This is not a valid counterexample."
                )
                if additional_info:
                    discussion += f"\n\nAdditional information: {' '.join(additional_info)}"
                return 0.0, "Not a valid counterexample", discussion

        except Exception as e:
            logger.error(f"Error scoring REFUTE problem: {e}")
            return 0.0, f"Scoring error: {str(e)}", None


if __name__ == "__main__":
    # Simple test to verify the benchmark loads correctly
    import asyncio

    async def test_refute():
        benchmark = RefuteBenchmark(seed=42, subset_size=5)
        print(f"Loaded {len(benchmark.problems)} problems")

        for i, problem in enumerate(benchmark.problems):
            print(f"\nProblem {i+1}: {problem.problem_id}")
            print(f"  Rating: {problem.answer.problem_rating}")
            print(f"  Tags: {problem.answer.tags}")

        return benchmark.problems[0] if benchmark.problems else None

    asyncio.run(test_refute())
