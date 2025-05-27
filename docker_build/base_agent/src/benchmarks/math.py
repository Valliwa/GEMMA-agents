# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import logging
import random

from pathlib import Path
from datasets import load_dataset
from dataclasses import dataclass
from typing import List, Tuple, Optional

from .base import BaseBenchmark, Problem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Interval:
    """Represents a mathematical interval"""
    start: float
    end: float
    left_closed: bool  # True if [ on left, False if (
    right_closed: bool  # True if ] on right, False if )


@dataclass
class Matrix:
    """Represents a matrix of numbers"""
    values: List[List[float]]


class SequenceList(list):
    pass


def parse_matrix(s: str) -> Optional[Matrix]:
    """Parse a LaTeX matrix into a Matrix object"""
    # Handle pmatrix, bmatrix, etc.
    if not (s.startswith("\\begin{") and s.endswith("}")):
        return None

    try:
        # Remove the matrix environment tags
        content = re.search(r"\\begin{[^}]+}(.*)\\end{[^}]+}", s)
        if not content:
            return None
        content = content.group(1)

        # Split into rows
        rows = content.strip().split("\\\\")
        matrix = []
        for row in rows:
            # Split row into elements and convert to floats
            elements = [float(x.strip()) for x in row.strip().split("&")]
            matrix.append(elements)

        return Matrix(values=matrix)
    except Exception:
        return None


def parse_interval(s: str) -> Optional[Interval]:
    """Parse an interval notation string into an Interval object"""
    try:
        # Match interval pattern [a,b] or (a,b) or mixed
        pattern = r'[\[\(]([-+]?\d*\.?\d+),([-+]?\d*\.?\d+)[\]\)]'
        match = re.match(pattern, s.strip())
        if not match:
            return None

        start = float(match.group(1))
        end = float(match.group(2))
        left_closed = s[0] == '['
        right_closed = s[-1] == ']'

        return Interval(start, end, left_closed, right_closed)
    except Exception:
        return None


def parse_union(s: str) -> Optional[List[Interval]]:
    """Parse a union of intervals"""
    try:
        if '\\cup' not in s:
            return None

        intervals = []
        for interval_str in s.split('\\cup'):
            interval = parse_interval(interval_str.strip())
            if interval is None:
                return None
            intervals.append(interval)

        return intervals
    except Exception:
        return None


def parse_sequence(s: str) -> list[float] | None:
    def get_sequence_type(s: str) -> str | None:
        if s.startswith("(") and s.endswith(")"):
            return "tuple"
        elif s.startswith("[") and s.endswith("]"):
            return "list"
        return None

    sequence_type = get_sequence_type(s)
    if sequence_type is None:
        return None

    try:
        content = s[1:-1].strip()

        elements = []
        start = 0
        i = 0
        while i < len(content):
            if content[i] == ',':
                next_chars = content[i+1:].lstrip()
                is_thousands = False
                if next_chars:
                    digits_after = 0
                    j = 0
                    while j < len(next_chars) and digits_after < 4:
                        if next_chars[j].isdigit():
                            digits_after += 1
                        elif next_chars[j].isspace():
                            pass
                        else:
                            break
                        j += 1
                    is_thousands = (digits_after == 3)

                if not is_thousands:
                    element = content[start:i].strip()
                    if element:
                        elements.append(element)
                    start = i + 1
            i += 1

        final_element = content[start:].strip()
        if final_element:
            elements.append(final_element)

        # Use the custom list subclass so we can attach an attribute.
        result = SequenceList([float(e.replace(",", "")) for e in elements])
        result.sequence_type = sequence_type  # Now this works.
        return result

    except Exception as e:
        return None


def parse_latex_fraction(s: str) -> float | None:
    """Parse a LaTeX fraction into a float"""
    try:
        if not s.startswith('\\'):  # Quick fail if not LaTeX
            return None

        # Handle \frac, \tfrac, \dfrac
        if not any(s.startswith(cmd) for cmd in ['\\frac', '\\tfrac', '\\dfrac']):
            return None

        # Standardize to \frac
        s = s.replace('tfrac', 'frac').replace('dfrac', 'frac')

        if not s.startswith('\\frac{'):
            # Handle \frac12 style
            if len(s) >= 6:  # \frac + at least 2 chars
                num = s[5]
                if len(s) > 6:
                    den = s[6]
                    return float(num) / float(den)
            return None

        # Handle \frac{...}{...} style
        content = s[6:]  # Remove \frac{
        brace_count = 1
        i = 0
        while i < len(content) and brace_count > 0:
            if content[i] == '{':
                brace_count += 1
            elif content[i] == '}':
                brace_count -= 1
            i += 1

        if brace_count > 0 or i >= len(content):
            return None

        numerator = content[:i-1]  # -1 to remove closing brace

        if not content[i:].startswith('{'):
            return None

        denominator = content[i+1:-1]  # Remove outer braces

        return float(numerator) / float(denominator)
    except Exception:
        return None

def convert_to_float(s: str) -> float | None:
    """Convert any numeric string to float, handling various formats."""
    s = s.strip()
    try:
        # Direct conversion (handles thousands separators)
        return float(s.replace(",", ""))
    except ValueError:
        # Try simple fraction
        if "/" in s and "\\" not in s:
            try:
                num, den = map(float, s.split("/"))
                return num / den
            except Exception:
                pass

        # Try LaTeX fraction
        latex_val = parse_latex_fraction(s)
        if latex_val is not None:
            return latex_val

    return None


def compare_numeric(a: str, b: str) -> bool:
    """Compare two strings numerically, handling numbers, sequences, and matrices."""
    # Try sequence comparison first
    a_seq = parse_sequence(a)
    b_seq = parse_sequence(b)

    if a_seq is not None and b_seq is not None:
        # Check that the sequence types match
        if getattr(a_seq, 'sequence_type', None) != getattr(b_seq, 'sequence_type', None):
            return False

        if len(a_seq) != len(b_seq):
            return False

        return all(abs(x - y) < 1e-6 for x, y in zip(a_seq, b_seq))

    # Try matrix comparison next
    a_matrix = parse_matrix(a)
    b_matrix = parse_matrix(b)
    if a_matrix is not None and b_matrix is not None:
        if len(a_matrix.values) != len(b_matrix.values):
            return False
        for row_a, row_b in zip(a_matrix.values, b_matrix.values):
            if len(row_a) != len(row_b):
                return False
            if any(abs(x - y) > 1e-6 for x, y in zip(row_a, row_b)):
                return False
        return True

    # Try direct numeric conversion
    a_val = convert_to_float(a)
    b_val = convert_to_float(b)
    if a_val is not None and b_val is not None:
        return abs(a_val - b_val) < 1e-6

    return False


def standardize_format(string: str) -> str | None:
    """Standardize LaTeX formatting only, without changing mathematical meaning."""
    if string is None:
        return None

    # Remove whitespace
    string = string.replace(" ", "")

    # Basic LaTeX cleanup
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac").replace("dfrac", "frac")
    string = string.replace("\\left", "").replace("\\right", "")
    string = string.replace("^{\\circ}", "").replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "").replace("\\%", "")
    string = string.replace("\n", "").replace("\\!", "")

    # Standardize sqrt notation
    parts = string.split("\\sqrt")
    if len(parts) > 1:
        result = parts[0]
        for part in parts[1:]:
            if part and part[0] != "{":
                first_char = part[0]
                rest = part[1:] if len(part) > 1 else ""
                result += f"\\sqrt{{{first_char}}}{rest}"
            else:
                result += "\\sqrt" + part
        string = result

    # Handle units and trailing text
    if "\\text{" in string:
        string = string.split("\\text{")[0]

    return string


@dataclass
class MATHExample:
    """A single MATH dataset example."""
    problem_id: str
    problem: str
    solution: str
    level: str  # Level 1-5
    type: str   # Subject type (Algebra, Geometry, etc)
    answer: str

    @classmethod
    def from_raw(cls, example: dict) -> "MATHExample":
        """Create a MATHExample from a raw dataset example."""
        solution = example["solution"]

        # Extract answer preferring boxed content
        if "\\boxed{" in solution:
            boxed_start = solution.find("\\boxed{") + 7
            boxed_end = solution.find("}", boxed_start)
            if boxed_end != -1:
                answer = solution[boxed_start:boxed_end]
            else:
                answer = solution
        elif "#### " in solution:
            answer = solution.split("#### ")[-1]
        elif "The answer is " in solution:
            answer = solution.split("The answer is ")[-1]
        else:
            answer = solution

        return cls(
            problem_id=str(example.get("id", "")),
            problem=example["problem"].strip(),
            solution=solution,
            level=str(example.get("level", "")),
            type=example.get("type", ""),
            answer=answer.strip()
        )


class MATHBenchmark(BaseBenchmark):
    """Benchmark for the Hendrycks MATH dataset."""

    name = "math"

    def __init__(self, seed: int | None = 1, subset_size: int | None = 20):
        super().__init__(seed, subset_size)

        # Load dataset
        dataset = load_dataset("hendrycks/competition_math", trust_remote_code=True)
        self.test_data = [MATHExample.from_raw(ex) for ex in dataset["test"]]

        # Create randomized subset if requested
        if subset_size is not None:
            random.seed(seed)
            self.test_data = random.sample(self.test_data, subset_size)

        # Convert to Problem instances
        self._data = [
            Problem(
                problem_id=ex.problem_id or str(i),
                statement=f"Problem Type: {ex.type}\nDifficulty Level: {ex.level}\n\n{ex.problem}",
                answer=ex.answer,
                answer_discussion=ex.solution,
            )
            for i, ex in enumerate(self.test_data)
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
    ) -> Tuple[float, str | None, str | None]:
        """Score the answer from the agent against the ground truth."""
        try:
            answer_path = Path(agent_answer_dir) / "answer.txt"
            agent_answer = answer_path.read_text().strip()

            # Standardize the formatting of both answers
            truth = standardize_format(problem.answer)
            pred = standardize_format(agent_answer)

            if truth is None or pred is None:
                return 0.0, "Invalid answer format", problem.answer_discussion

            # First try exact string match after formatting standardization
            if truth == pred:
                return 1.0, None, problem.answer_discussion

            # Then try numeric comparison (handles both single numbers and sequences)
            if compare_numeric(truth, pred):
                return 1.0, None, problem.answer_discussion

            return 0.0, None, problem.answer_discussion

        except Exception as e:
            return 0.0, str(e), problem.answer_discussion

    def filter_by_level(self, level: int) -> list[Problem]:
        """Filter problems by difficulty level (1-5)."""
        return [
            p
            for i, p in enumerate(self.problems)
            if self.test_data[i].level == str(level)
        ]

    def filter_by_type(self, problem_type: str) -> list[Problem]:
        """Filter problems by type (e.g., 'Algebra', 'Geometry')."""
        return [
            p
            for i, p in enumerate(self.problems)
            if self.test_data[i].type == problem_type
        ]

def analyze_dataset_answers(problems: list[Problem], sample_size: int = 100) -> None:
    """Analyze a sample of answers from the dataset to understand the types of answers we need to handle."""
    if len(problems) > sample_size:
        problems = random.sample(problems, sample_size)

    print(f"\nAnalyzing {len(problems)} answers from the dataset:")
    print("=" * 80)

    answer_types = {}
    for prob in problems:
        # Try to categorize the answer
        ans = prob.answer
        if ans is None:
            category = "None"
        elif ans.startswith("\\frac"):
            category = "LaTeX Fraction"
        elif any(ans.startswith(d) for d in "(["):
            category = "Sequence"
        elif "\\sqrt" in ans:
            category = "Square Root"
        elif ans.replace(".", "").replace(",", "").isdigit():
            category = "Simple Number"
        elif "/" in ans and "\\" not in ans:
            category = "Simple Fraction"
        elif any(op in ans for op in "+-*/"):
            category = "Expression"
        else:
            category = "Other"

        answer_types[category] = answer_types.get(category, []) + [ans]

    # Print summary
    print("\nAnswer Type Distribution:")
    print("-" * 40)
    for category, examples in answer_types.items():
        print(f"\n{category} ({len(examples)} instances):")
        print("Example answers:")
        for ex in examples[:3]:  # Show up to 3 examples
            print(f"  {ex}")

    print("\nNote: This is a random sample and may not represent all possible answer types.")


if __name__ == "__main__":
    import tempfile
    import os

    def run_test_case(benchmark: MATHBenchmark, answer_dir: Path,
                      ground_truth: str, agent_answer: str, should_pass: bool):
        """Helper function to run a single test case"""
        print(f"\nTESTING: '{ground_truth}' vs '{agent_answer}' (should_pass={should_pass})")

        problem = benchmark.problems[0]  # Use first problem as template
        problem.answer = ground_truth
        problem.answer_discussion = "Test discussion"

        answer_file = answer_dir / "answer.txt"
        answer_file.write_text(agent_answer)

        score, _, _ = benchmark.score_problem(
            problem, str(answer_dir.parent), str(answer_dir), "test"
        )

        assert score == (1.0 if should_pass else 0.0), \
            f"Failed: '{ground_truth}' vs '{agent_answer}' got {score}, expected {1.0 if should_pass else 0.0}"

    # Create test environment
    benchmark = MATHBenchmark(seed=42, subset_size=10)

    with tempfile.TemporaryDirectory() as tmpdir:
        answer_dir = Path(tmpdir) / "answers"
        answer_dir.mkdir()

        print("\nTesting basic strings...")
        test_cases = [
            ("abc", "abc", True),
            ("abc", "def", False),
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        print("\nTesting LaTeX fractions...")
        test_cases = [
            ("\\frac{1}{2}", "\\frac{1}{2}", True),
            ("\\tfrac{1}{2}", "\\frac{1}{2}", True),
            ("\\dfrac{1}{2}", "\\frac{1}{2}", True),
            ("\\frac{2}{4}", "0.5", True),
            ("\\frac{49}{7}", "7", True),
            ("\\frac{1}{3}", "0.33", False),  # Not equal within epsilon
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        print("\nTesting LaTeX formatting...")
        test_cases = [
            ("x + y", "x+y", True),
            ("\\sqrt{2}", "\\sqrt2", True),
            ("10\\text{ units}", "10", True),
            ("\\$10", "10", True),
            ("10\\%", "10", True),
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        print("\nTesting numeric equivalence...")
        test_cases = [
            ("1", "1", True),
            ("1", "2", False),
            ("1.0", "1", True),
            ("1.00", "1", True),
            ("1.00001", "1", False),
            ("1/2", "0.5", True),
            ("2/4", "0.5", True),
            ("3/4", "0.7", False),
            ("1,000", "1000", True),
            ("1,000.0", "1000", True),
            ("-256", "-256.0", True),
            ("-256.01", "-256", False),
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        print("\nTesting sequences (tuples and lists)...")
        test_cases = [
            ("(1, 2, 3)", "(1,2,3)", True),
            ("(1.0, 2.0, 3)", "(1, 2, 3)", True),
            ("(-3, -2, 9)", "(-3,-2,9)", True),
            ("(-3, -2, 9)", "(-3, -2, 9.0)", True),
            ("(1, 2)", "(1, 2, 3)", False),
            ("(1, 2, 3)", "(3, 2, 1)", False),
            ("(1.5, -2.0)", "(1.5,-2)", True),
            ("(1,000, 2,000)", "(1000,2000)", True),
            ("[1, 2, 3]", "[1,2,3]", True),
            ("[1.0, 2.0, 3]", "[1, 2, 3]", True),
            ("[-3, -2, 9]", "[-3,-2,9]", True),
            ("[-3, -2, 9]", "[-3, -2, 9.0]", True),
            ("[1, 2]", "[1, 2, 3]", False),
            ("[1, 2, 3]", "[3, 2, 1]", False),
            ("[1.5, -2.0]", "[1.5,-2]", True),
            ("[1,000, 2,000]", "[1000,2000]", True),
            # Mixed types should not be equal
            ("(1, 2, 3)", "[1, 2, 3]", False),
            ("[1.0, 2.0]", "(1, 2)", False),
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        print("\nTesting intervals and unions...")
        test_cases = [
            ("[1,2]", "[1,2]", True),
            ("(1,2)", "(1,2)", True),
            ("[1,2)", "[1,2)", True),
            ("[1,2]", "(1,2)", False),
            ("[1,2]", "[1,3]", False),
            ("[1,2] \\cup [3,4]", "[1,2] \\cup [3,4]", True),
            ("[-1,0] \\cup [2,3]", "[-1,0] \\cup [2,3]", True),
            ("[1,2] \\cup [3,4]", "[1,2] \\cup [3,5]", False),
            ("[-3,-2] \\cup [0,1]", "[-3,-2] \\cup [0,1]", True),
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        print("\nTesting matrices...")
        test_cases = [
            ("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}",
             "\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}", True),
            ("\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}",
             "\\begin{pmatrix}1&2\\\\3&5\\end{pmatrix}", False),
            # TODO: fix this case
            # ("\\begin{pmatrix}1.0&2.0\\\\3&4\\end{pmatrix}",
            #  "\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}", True),
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        print("\nTesting units and currency...")
        test_cases = [
            ("5\\text{ meters}", "5", True),
            ("\\$7.50", "7.50", True),
            ("10\\text{ seconds}", "10", True),
            ("45\\text{ degrees}", "45", True),
            ("\\$1,234.56", "1234.56", True),
        ]
        for truth, pred, should_pass in test_cases:
            run_test_case(benchmark, answer_dir, truth, pred, should_pass)

        # Optional: analyze answers from a larger dataset
        print("\nAnalyzing dataset answers...")
        benchmark_large = MATHBenchmark(seed=42, subset_size=2000)
        analyze_dataset_answers(benchmark_large.problems)

        print("\nAll tests passed! ✨")

    # benchmark = MATHBenchmark(seed=42, subset_size=2000)
    # analyze_dataset_answers(benchmark.problems, 2000)
