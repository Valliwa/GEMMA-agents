# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import time
import json
import shutil
import random
import base64
import difflib
import logging
import subprocess

from typing import List, Dict, Optional, ClassVar, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from diff_match_patch import diff_match_patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import git

from .base import BaseBenchmark, Problem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Standard set of repositories to analyze
REPO_LIST: Set[str] = {
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pydata/xarray",
    "pylint-dev/pylint",
    "pytest-dev/pytest",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
}

@dataclass
class FileEditProblem(Problem):
    """Problem subclass specifically for file editing tasks"""

    repo_name: str
    filepath: str
    base_commit: str
    target_commit: str
    base_content: str  # base64 encoded
    commit_message: str
    commit_date: str
    author: str
    diff_stats: dict

    @classmethod
    def from_file_edit(cls, edit_dict: Dict, problem_id: str) -> "FileEditProblem":
        """Create a FileEditProblem from a file edit dictionary"""
        # Clean the repository name and commit to create a safe problem ID
        repo_base = edit_dict["repo_name"].split("/")[-1]
        safe_id = re.sub(
            r"[^a-zA-Z0-9_]", "", f"{repo_base}_{edit_dict['target_commit'][:8]}"
        )

        # Decode the target content that we want to achieve
        target_content = base64.b64decode(edit_dict["target_content"]).decode("utf-8")

        repo_dir = edit_dict["repo_name"].split("/")[-1]
        filepath = f"{repo_dir}/{edit_dict['filepath']}"

        # Create the problem statement
        statement = (
            f"Your task is to efficiently edit the file at {filepath} so that its content is:\n\n"
            f"<TARGET_CONTENT>\n{target_content}\n</TARGET_CONTENT>"
            f"\n\n"
            f"For context, the goal of this edit was: `{edit_dict['commit_message']}`\n\n"
            f"Please edit the file at {filepath} so that its contents exactly matches the provided target content, and then simply return 'done' as your answer.\n"
            f"Avoid writing out very long files verbatim."
        )

        return cls(
            problem_id=safe_id,
            statement=statement,
            answer=edit_dict["target_content"],  # base64 encoded target content
            answer_discussion=None,
            repo_name=edit_dict["repo_name"],
            filepath=edit_dict["filepath"],
            base_commit=edit_dict["base_commit"],
            target_commit=edit_dict["target_commit"],
            base_content=edit_dict["base_content"],
            commit_message=edit_dict["commit_message"],
            commit_date=edit_dict["commit_date"],
            author=edit_dict["author"],
            diff_stats=edit_dict["diff_stats"],
        )

def compute_diff_similarity(base_content: str, target_content: str, edited_content: str) -> Tuple[float, Optional[str]]:
    """
    Compute similarity score by analyzing the accuracy of changes made.
    The score is based on:
    1. How many of the required changes were made correctly
    2. Penalties for incorrect changes
    3. Weighting that considers the significance of changes

    Returns (score, message) tuple where score is in [0, 1].
    """
    try:
        # Quick equality tests first
        if target_content == edited_content:
            return (1.0, None)
        if base_content == edited_content:
            return (0.0, "No changes were made when changes were required")

        def normalize_content(content: str) -> List[str]:
            """Normalize content and return as list of lines with consistent line endings"""
            lines = [line.rstrip() for line in content.replace('\r\n', '\n').split('\n')]
            return [line for line in lines if line]  # Remove empty lines

        # Normalize all content
        base_lines = normalize_content(base_content)
        target_lines = normalize_content(target_content)
        edited_lines = normalize_content(edited_content)

        # Use sequence matcher for accurate line-based diffing
        target_matcher = difflib.SequenceMatcher(None, base_lines, target_lines)
        edited_matcher = difflib.SequenceMatcher(None, base_lines, edited_lines)

        # Get detailed change information
        target_opcodes = target_matcher.get_opcodes()
        edited_opcodes = edited_matcher.get_opcodes()

        # Track required vs actual changes
        required_changes = set()  # Lines that need to be changed
        actual_changes = set()    # Lines that were changed
        correct_changes = set()   # Changes that match target

        # Analyze required changes
        for tag, i1, i2, j1, j2 in target_opcodes:
            if tag != 'equal':
                # Add all changed line indices to required set
                required_changes.update(range(j1, j2))

                # For Replace operations, also mark original lines
                if tag == 'replace':
                    required_changes.update(range(i1, i2))

        # Analyze actual changes
        for tag, i1, i2, j1, j2 in edited_opcodes:
            if tag != 'equal':
                # Track which lines were modified
                actual_changes.update(range(j1, j2))

                # Check if changes match target
                for idx in range(j1, j2):
                    if idx < len(edited_lines) and idx < len(target_lines):
                        if edited_lines[idx] == target_lines[idx]:
                            correct_changes.add(idx)

        # Calculate metrics
        num_required = len(required_changes) or 1   # Avoid division by zero
        num_actual = len(actual_changes) or 1
        num_correct = len(correct_changes)

        # Calculate component scores
        precision = num_correct / num_actual   # How accurate were the changes
        recall = num_correct / num_required    # How complete were the changes

        # Calculate excess change penalty
        if num_actual > num_required:
            excess_ratio = (num_actual - num_required) / num_required
            excess_penalty = min(0.3, 0.1 * excess_ratio)  # Cap at 0.3
        else:
            excess_penalty = 0

        # No changes means zero score
        if num_correct == 0:
            return (0.0, f"Score: 0.00 - Made {num_actual} changes, none correct")

        # Calculate weighted score
        weight_precision = 0.4  # Emphasize precision slightly less
        weight_recall = 0.6     # Emphasize recall slightly more

        weighted_score = (
            weight_precision * precision +
            weight_recall * recall
        )

        # Apply penalties and ensure non-negative
        final_score = max(0.0, weighted_score - excess_penalty)

        # Generate detailed message
        msg_parts = []
        msg_parts.append(f"Correct changes: {num_correct}/{num_required}")
        msg_parts.append(f"Precision: {precision:.2f}")
        msg_parts.append(f"Recall: {recall:.2f}")
        if excess_penalty > 0:
            msg_parts.append(f"Excess changes penalty: {excess_penalty:.2f}")

        msg = f"Score: {final_score:.2f} - " + ", ".join(msg_parts)

        return (final_score, msg)

    except Exception as e:
        return (0.0, f"Error computing similarity: {str(e)}")




def compute_diff_similarity_old(base_content: str, target_content: str, edited_content: str) -> Tuple[float, Optional[str]]:
    """
    Compute similarity score between target and edited content using diff-match-patch.
    Optimized for large files with timeout protection.
    Returns a tuple of (score, error_message).
    """
    try:
        def normalize_content(content: str) -> str:
            """Normalize content by:
            1. Converting line endings to \n
            2. Removing trailing whitespace from each line
            3. Removing empty lines at the end of the file
            4. Ensuring exactly one newline at the end
            """
            lines = content.replace('\r\n', '\n').split('\n')
            lines = [line.rstrip() for line in lines]
            while lines and not lines[-1]:
                lines.pop()
            return '\n'.join(lines) + '\n'

        # First do a quick equality check before any normalization
        if target_content == edited_content:
            return (1.0, None)

        # Normalize all content
        base_norm = normalize_content(base_content)
        target_norm = normalize_content(target_content)
        edited_norm = normalize_content(edited_content)

        # Check equality after normalization
        if target_norm == edited_norm:
            return (1.0, None)

        # If no changes were needed
        if base_norm == target_norm:
            return (1.0, None) if base_norm == edited_norm else (0.0, "Changes made when none were needed")

        # Initialize diff-match-patch with increased timeout for large files
        dmp = diff_match_patch()
        dmp.Diff_Timeout = 5.0  # 5 seconds

        # For large files, first try line-by-line diff
        if len(target_norm) > 50000:  # For files > 50KB
            target_lines = target_norm.splitlines()
            edited_lines = edited_norm.splitlines()

            # Quick comparison of line counts
            if len(target_lines) == len(edited_lines):
                matching_lines = sum(1 for t, e in zip(target_lines, edited_lines) if t == e)
                if matching_lines == len(target_lines):
                    return (1.0, None)

        # Time the diff operation
        start_time = time.time()

        try:
            diffs = dmp.diff_main(target_norm, edited_norm)
            diff_time = time.time() - start_time

            # Only do semantic cleanup for smaller files or quick diffs
            if len(target_norm) < 50000 or diff_time < 1.0:
                dmp.diff_cleanupSemantic(diffs)

            # Calculate similarity score
            matching_chars = sum(len(text) for op, text in diffs if op == 0)
            total_chars = sum(len(text) for op, text in diffs)

            if total_chars == 0:
                return (1.0, None)

            score = matching_chars / total_chars

            # Generate detailed message for smaller files
            if len(target_norm) < 10000:
                diff_details = []
                for op, text in diffs:
                    if op == 1:  # Insertion
                        diff_details.append(f"Added: '{text.strip()}'")
                    elif op == -1:  # Deletion
                        diff_details.append(f"Missing: '{text.strip()}'")

                msg = f"Content differs. Similarity: {score:.2f}. " + ", ".join(diff_details[:3])
                if len(diff_details) > 3:
                    msg += f" (and {len(diff_details) - 3} more differences)"
            else:
                msg = f"Content differs. Similarity: {score:.2f} (diff took {diff_time:.1f}s)"

            return (score, msg)

        except Exception as e:
            if time.time() - start_time >= dmp.Diff_Timeout:
                return (0.0, f"Diff timeout after {dmp.Diff_Timeout}s")
            raise

    except Exception as e:
        return (0.0, f"Error computing similarity: {str(e)}")


class FileEditingBenchmark(BaseBenchmark):
    """Benchmark for evaluating file editing capabilities"""

    name: ClassVar[str] = "file_editing"

    def __init__(self, seed: int = 1, subset_size: int = 5):
        super().__init__(seed, subset_size)
        self.seed = seed
        self.subset_size = subset_size

        # Use the new conventional benchmark data location
        self._dataset_path = (
            Path(__file__).parents[3]  # Go up to project root
            / "benchmark_data"
            / "file_editing_bench"
            / "file_edits.json"
        )

        self._problems: Optional[List[FileEditProblem]] = None
        self._problem_map: Dict[str, FileEditProblem] = {}
        self._problems_shuffled = False

        # Initialize benchmark data if missing
        self._ensure_benchmark_data()

    def _clone_repository(self, repo_name: str, repos_dir: Path) -> Optional[git.Repo]:
        """Clone a repository if it doesn't exist, or fetch updates if it does"""
        repo_dir = repos_dir / repo_name.split("/")[-1]

        try:
            if repo_dir.exists():
                logger.info(f"Repository {repo_name} exists, fetching updates...")
                repo = git.Repo(repo_dir)
                repo.remotes.origin.fetch()
                return repo
            else:
                logger.info(f"Cloning repository {repo_name}...")
                return git.Repo.clone_from(
                    f"https://github.com/{repo_name}.git",
                    repo_dir,
                    depth=None,  # Full clone to access history
                )
        except Exception as e:
            logger.error(f"Error processing repository {repo_name}: {str(e)}")
            return None

    def _get_default_branch(self, repo: git.Repo) -> str:
        """Get the name of the default branch (main or master)"""
        try:
            # Try to get the branch the remote HEAD points to
            default_branch = repo.git.symbolic_ref("refs/remotes/origin/HEAD").replace(
                "refs/remotes/origin/", ""
            )
            return default_branch
        except git.exc.GitCommandError:
            # If that fails, check if main or master exists
            remote_refs = [ref.name for ref in repo.remote().refs]
            if "origin/main" in remote_refs:
                return "main"
            elif "origin/master" in remote_refs:
                return "master"
            else:
                # Get the first branch as fallback
                return remote_refs[0].replace("origin/", "") if remote_refs else None

    def _get_file_content(self, repo: git.Repo, commit: str, filepath: str) -> Optional[str]:
        """Get content of a file at a specific commit and return as base64"""
        try:
            # Get raw content as bytes directly from git
            content_bytes = repo.git.show(f"{commit}:{filepath}").encode("utf-8")

            # Check if content is too large (e.g., >1MB)
            if len(content_bytes) > 1_000_000:
                logger.warning(
                    f"File {filepath} at {commit} is too large ({len(content_bytes)} bytes), skipping"
                )
                return None

            # Convert to base64
            return base64.b64encode(content_bytes).decode("ascii")

        except git.exc.GitCommandError:
            return None
        except Exception as e:
            logger.warning(f"Error reading {filepath} at {commit}: {e}")
            return None

    def _get_file_edits(self, repo: git.Repo, repo_name: str, max_edits: int = 100) -> List[Dict]:
        """Extract file edit examples from repository history"""
        edits = []

        try:
            # Get the default branch
            default_branch = self._get_default_branch(repo)
            if not default_branch:
                logger.error(f"Could not determine default branch for {repo_name}")
                return edits

            logger.info(f"Using branch '{default_branch}' for {repo_name}")

            # Get commit history
            commits = list(repo.iter_commits(default_branch))

            for commit in commits:
                if len(edits) >= max_edits:
                    break

                # Skip merge commits and commits with multiple file changes
                if len(commit.parents) != 1 or len(commit.stats.files) != 1:
                    continue

                # Get the single modified file
                modified_file = next(iter(commit.stats.files))

                # Skip if not a Python file
                if not modified_file.endswith(".py"):
                    continue

                # Get file content at both commits
                target_content = self._get_file_content(repo, commit.hexsha, modified_file)
                base_content = self._get_file_content(
                    repo, commit.parents[0].hexsha, modified_file
                )

                # Skip if we couldn't get either content version
                if not target_content or not base_content:
                    continue

                # Create edit dictionary
                edit = {
                    "repo_name": repo_name,
                    "filepath": modified_file,
                    "base_commit": str(commit.parents[0].hexsha),
                    "target_commit": str(commit.hexsha),
                    "base_content": base_content,
                    "target_content": target_content,
                    "commit_message": commit.message.strip(),
                    "commit_date": commit.committed_datetime.isoformat(),
                    "author": f"{commit.author.name} <{commit.author.email}>",
                    "diff_stats": commit.stats.files[modified_file],
                }

                edits.append(edit)
                logger.debug(
                    f"Added edit for {modified_file} from commit {commit.hexsha[:8]}"
                )

        except Exception as e:
            logger.error(f"Error processing commits for {repo_name}: {str(e)}")

        return edits

    def _save_dataset(self, edits: List[Dict], output_file: Path):
        """Save the curated dataset to a JSON file"""
        # Convert to dict and ensure all strings are valid JSON
        dataset = []
        for edit in edits:
            try:
                # Test JSON serialization of each edit
                json.dumps(edit)
                dataset.append(edit)
            except Exception as e:
                logger.error(f"Failed to serialize edit: {str(e)}")
                continue

        try:
            # Write with ensure_ascii=False to preserve Unicode, but handle encoding
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with output_file.open("w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(dataset)} examples to {output_file}")
        except Exception as e:
            logger.error(f"Failed to save dataset: {str(e)}")
            # Try fallback with ASCII encoding if Unicode fails
            with output_file.open("w", encoding="ascii") as f:
                json.dump(dataset, f, indent=2, ensure_ascii=True)
            logger.info(f"Saved {len(dataset)} examples to {output_file} (ASCII-only)")

    def _curate_dataset(self, max_edits_per_repo: int = 100):
        """Create the benchmark dataset by processing repositories"""
        # Create directory structure
        repos_dir = self._dataset_path.parent / "repos"
        repos_dir.mkdir(parents=True, exist_ok=True)

        # Clone/update repositories in parallel
        all_edits = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_repo = {
                executor.submit(self._clone_repository, repo_name, repos_dir): repo_name
                for repo_name in REPO_LIST
            }

            for future in as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                try:
                    repo = future.result()
                    if repo:
                        # Extract file edits from repository
                        edits = self._get_file_edits(repo, repo_name, max_edits_per_repo)
                        all_edits.extend(edits)
                        logger.info(f"Extracted {len(edits)} examples from {repo_name}")
                except Exception as e:
                    logger.error(f"Error processing {repo_name}: {str(e)}")

        # Save the dataset
        self._save_dataset(all_edits, self._dataset_path)

    def _ensure_benchmark_data(self):
        """Ensure benchmark data exists, creating it if necessary"""
        if not self._dataset_path.exists():
            logger.info("Benchmark data not found, generating...")
            self._curate_dataset()

    def _shuffle_problems(self) -> None:
        """Shuffle the problems list using the specified seed"""
        if self._problems and not self._problems_shuffled:
            random.seed(self.seed)
            random.shuffle(self._problems)
            self._problems_shuffled = True

    @property
    def problems(self) -> List[Problem]:
        """Get a subset of shuffled problems from the benchmark"""
        if self._problems is None:
            self._load_problems()

        # Shuffle problems if not already shuffled
        self._shuffle_problems()

        # Return subset of problems based on subset_size
        return self._problems[: min(self.subset_size, len(self._problems))]

    def get_problem(self, problem_id: str) -> Optional[Problem]:
        """Get a specific problem by ID"""
        if self._problems is None:
            self._load_problems()

        return self._problem_map.get(problem_id)

    def _load_problems(self) -> None:
        """Load problems from the dataset file"""
        try:
            with open(self._dataset_path, "r") as f:
                dataset = json.load(f)

            self._problems = []
            for edit in dataset:
                # Create problem ID combining repo name and commit
                problem_id = f"{edit['repo_name']}_{edit['target_commit'][:8]}"

                problem = FileEditProblem.from_file_edit(edit, problem_id)

                if len(base64.b64decode(problem.answer).decode("utf-8").splitlines()) < 1000:
                    self._problems.append(problem)
                    self._problem_map[problem_id] = problem

        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset from {self._dataset_path}: {str(e)}"
            )

    async def setup_problem(
        self, problem: Problem, problem_data_dir: Path, container_name: str
    ) -> None:
        """Setup the git repository for this problem."""
        if not isinstance(problem, FileEditProblem):
            raise ValueError("Problem must be a FileEditProblem instance")

        file_edit_problem: FileEditProblem = problem

        # Get repo name from the full repo path
        repo_name = file_edit_problem.repo_name.split("/")[-1]

        # Source repository path (from benchmark data directory)
        src_repo_path = self._dataset_path.parent / "repos" / repo_name
        if not src_repo_path.exists():
            raise RuntimeError(f"Source repository not found at {src_repo_path}")

        dest_repo_path = problem_data_dir / repo_name

        # If the directory exists (from a previous run), remove it first
        if dest_repo_path.exists():
            shutil.rmtree(dest_repo_path)

        # Copy the entire repository first
        shutil.copytree(src_repo_path, dest_repo_path)

        try:
            # Clean any untracked files
            subprocess.run(
                ["git", "clean", "-fdx"],
                cwd=dest_repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

            # Reset any changes
            subprocess.run(
                ["git", "reset", "--hard"],
                cwd=dest_repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

            # Checkout the base commit
            subprocess.run(
                ["git", "checkout", file_edit_problem.base_commit],
                cwd=dest_repo_path,
                check=True,
                capture_output=True,
                text=True,
            )

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to execute git command: {e.cmd}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to checkout base commit: {str(e)}")

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> Tuple[float, str | None, str | None]:
        """Score the agent's file edit by comparing the diffs."""
        if not isinstance(problem, FileEditProblem):
            raise ValueError("Problem must be a FileEditProblem instance")

        file_edit_problem: FileEditProblem = problem

        try:
            # Get the base and target content
            base_content = base64.b64decode(file_edit_problem.base_content).decode(
                "utf-8"
            )
            target_content = base64.b64decode(file_edit_problem.answer).decode("utf-8")

            # Get the edited file content from the workdir
            repo_name = file_edit_problem.repo_name.split("/")[-1]
            edited_file_path = (
                Path(agent_workdir) / repo_name / file_edit_problem.filepath
            )

            if not edited_file_path.exists():
                return 0.0, "Edited file not found", None

            # Read the edited content
            with open(edited_file_path, "r", encoding="utf-8") as f:
                edited_content = f.read()

            # Normalize line endings
            base_content = base_content.replace("\r\n", "\n")
            target_content = target_content.replace("\r\n", "\n")
            edited_content = edited_content.replace("\r\n", "\n")

            # Compute similarity score
            score, warnings = compute_diff_similarity(
                base_content, target_content, edited_content
            )
            return score, warnings, None

        except Exception as e:
            return 0.0, f"Error scoring file edit: {str(e)}", None


if __name__ == '__main__':
    def generate_large_file(num_lines: int) -> str:
        """Generate a synthetic Python-like file with given number of lines"""
        lines = []
        indent_level = 0

        for i in range(num_lines):
            # Adjust indent level occasionally
            if random.random() < 0.3:
                if random.random() < 0.5 and indent_level > 0:
                    indent_level -= 1
                else:
                    indent_level += 1

            # Generate a line with current indentation
            line_types = [
                f"def function_{i}():",
                f"x_{i} = {random.randint(1, 100)}",
                f"result_{i} = calculate(param_{i})",
                f"if condition_{i}:",
                f"for item_{i} in list_{i}:",
                f"# Comment line {i}",
                f"print(f'Debug message {i}')",
                f"return value_{i}"
            ]

            line = "    " * indent_level + random.choice(line_types)
            lines.append(line)

        return '\n'.join(lines) + '\n'

    def make_random_changes(content: str, change_ratio: float = 0.05) -> str:
        """Make random changes to about change_ratio of the lines"""
        lines = content.split('\n')
        num_changes = int(len(lines) * change_ratio)

        for _ in range(num_changes):
            idx = random.randint(0, len(lines) - 1)
            change_type = random.choice(['modify', 'insert', 'delete'])

            if change_type == 'modify':
                lines[idx] = lines[idx] + "  # Modified"
            elif change_type == 'insert':
                indent = lines[idx].count('    ') * '    '
                lines.insert(idx, indent + f"# New line {idx}")
            elif change_type == 'delete':
                lines.pop(idx)

        return '\n'.join(lines) + '\n'

    # Run stress tests
    file_sizes = [1000, 3000, 5000, 10000]
    change_ratios = [0.01, 0.05, 0.10]  # 1%, 5%, and 10% changes

    for size in file_sizes:
        print(f"\nTesting with {size} lines:")
        base_content = generate_large_file(size)
        print(f"Base file size: {len(base_content):,} bytes")

        for ratio in change_ratios:
            print(f"\n  Change ratio: {ratio:.0%}")
            target_content = make_random_changes(base_content, ratio)
            print(f"  Target file size: {len(target_content):,} bytes")

            # Test cases
            print("  Test cases:")

            # 1. Perfect edit (should be fast due to equality check)
            start_time = time.time()
            score, msg = compute_diff_similarity(base_content, target_content, target_content)
            perfect_time = time.time() - start_time
            print(f"    Perfect edit: Score={score:.2f}, Time={perfect_time:.3f}s")
            if msg:
                print(f"      Message: {msg}")

            # 2. Slightly off edit (additional small changes)
            start_time = time.time()
            slightly_off = make_random_changes(target_content, 0.01)
            score, msg = compute_diff_similarity(base_content, target_content, slightly_off)
            slight_time = time.time() - start_time
            print(f"    Slightly off: Score={score:.2f}, Time={slight_time:.3f}s")
            if msg:
                print(f"      Message: {msg}")

            # 3. No changes made (should detect this quickly)
            start_time = time.time()
            score, msg = compute_diff_similarity(base_content, target_content, base_content)
            no_change_time = time.time() - start_time
            print(f"    No changes: Score={score:.2f}, Time={no_change_time:.3f}s")
            if msg:
                print(f"      Message: {msg}")
