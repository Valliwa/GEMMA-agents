# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
AIQ Project-Specific Benchmark Implementations

This module defines specialized benchmark classes for each project in the AIQ protocol.
Each benchmark focuses on a single project type but can generate multiple instances
of that project for more reliable performance evaluation.
"""

import json
import logging
import subprocess
import asyncio
import shutil
import tarfile
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple

from .base import BaseBenchmark, Problem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# List of implemented projects in the AIQ protocol
PROJECT_LIST = ["linalg", "csv_parsing", "messaging_app", "dist_kv_store"]


@dataclass
class AIQProjectProblem(Problem):
    """
    Problem subclass for the AIQ project-specific benchmarks

    Note that these problems are a little different from the standard benchmark
    problems in that there is no definitive answer; merely a score from the
    quality metric.
    """
    project_name: str


class BaseProjectAIQBenchmark(BaseBenchmark):
    """
    Base class for project-specific AIQ benchmarks.

    This class directly implements the abstract methods from BaseBenchmark
    required to support AIQ protocol evaluation, focusing on a single project type.
    """

    name: ClassVar[str] = "base_project_aiq"
    project_name: ClassVar[str] = None

    def __init__(self, seed: int = 2, subset_size: int | None = None):
        """
        Initialize a project-specific AIQ benchmark

        Args:
            seed: Random seed for consistent problem generation
            subset_size: Number of problem instances to generate (default: None = 1)
        """
        super().__init__(seed=seed, subset_size=subset_size)

        if self.project_name is None:
            raise ValueError("project_name must be defined in the subclass")

        if self.project_name not in PROJECT_LIST:
            raise ValueError(f"Unknown project: {self.project_name}. Must be one of {PROJECT_LIST}")

        self._all_problems = []

        # Create the specified number of problem instances
        # Each instance is the same project, but with a unique problem_id
        instances_count = subset_size if subset_size is not None else 1

        for i in range(instances_count):
            problem_id = f"{self.project_name}_{i+1}"
            self._all_problems.append(
                AIQProjectProblem(
                    problem_id=problem_id,
                    statement=f"Following the AIQ v5.0 protocol, improve the {self.project_name} project as much as you can within the budget. "
                             f"Run the quality.py script to measure your progress and use its CLI options for analysis.",
                    answer=None,  # No predefined answer
                    answer_discussion=None,
                    project_name=self.project_name,
                )
            )

    @property
    def problems(self) -> List[Problem]:
        """Return list of AIQ problems for this specific project"""
        return self._all_problems

    async def setup_problem(
        self,
        problem: Problem,
        problem_data_dir: Path,
        container_name: str
    ) -> None:
        """
        Set up the AIQ project in the container's workspace

        This method handles extracting tarballed project directories and
        establishing the initial quality baseline.

        Args:
            problem: The problem to set up
            problem_data_dir: Path to the problem data directory, which is mounted to /home/agent/workdir
            container_name: Name of the Docker container
        """
        if not isinstance(problem, AIQProjectProblem):
            raise ValueError("Problem must be an AIQProjectProblem instance")

        aiq_problem: AIQProjectProblem = problem

        try:
            # Define source project tarball path
            projects_dir = Path(__file__).parents[3] / "benchmark_data" / "aiq_bench"
            tarball_path = projects_dir / f"{aiq_problem.project_name}.tar.gz"

            if not tarball_path.exists():
                raise RuntimeError(f"Project tarball not found: {tarball_path}")

            # Create a temporary directory for extraction to better handle nested directories
            temp_extract_dir = Path("/tmp") / f"aiq_extract_{aiq_problem.project_name}_{int(time.time())}"
            os.makedirs(temp_extract_dir, exist_ok=True)
            logger.debug(f"Created temporary extraction directory: {temp_extract_dir}")

            # Extract the tarball to the temporary directory
            logger.debug(f"Extracting {tarball_path} to {temp_extract_dir}")
            with tarfile.open(tarball_path, "r:gz") as tar:
                tar.extractall(path=temp_extract_dir)

            # Find the actual project directory in the extracted content
            # This handles both cases:
            # 1. The tarball contains a single directory named after the project
            # 2. The tarball contains files directly without a container directory
            extracted_items = list(temp_extract_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                # Case 1: Single directory in tarball
                project_dir = extracted_items[0]
                logger.debug(f"Found single project directory: {project_dir.name}")
            else:
                # Case 2: Files directly in tarball root
                project_dir = temp_extract_dir
                logger.debug(f"Using all extracted files as project directory")

            # Copy all content from the identified project directory to the problem_data_dir
            logger.debug(f"Copying {project_dir} content to {problem_data_dir}")
            for item in project_dir.iterdir():
                target_path = problem_data_dir / item.name
                if item.is_dir():
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.copytree(item, target_path)
                else:
                    shutil.copy(item, target_path)

            # Clean up the temporary directory
            shutil.rmtree(temp_extract_dir)
            logger.debug(f"Cleaned up temporary extraction directory")

            # Make all scripts executable
            self._make_scripts_executable(problem_data_dir)

            # Run initial quality check to establish baseline
            logger.info(f"Running initial quality check in container {container_name}")
            success, stdout, stderr = await self._run_quality_command(
                container_name=container_name,
                command_type="standard",
                output_file="/home/agent/workdir/initial_quality.json"
            )

            if not success:
                logger.warning(f"Initial quality check failed: {stderr}")
                logger.warning(f"Stdout: {stdout}")

            # Also run with --aggregate-only for a clean baseline reference
            logger.debug(f"Running initial quality check with --aggregate-only in container {container_name}")
            agg_success, agg_stdout, agg_stderr = await self._run_quality_command(
                container_name=container_name,
                command_type="aggregate",
                output_file="/home/agent/workdir/initial_aggregate.json"
            )

            if not agg_success:
                logger.warning(f"Initial aggregate quality check failed: {agg_stderr}")

        except Exception as e:
            logger.exception(f"Error setting up AIQ problem: {e}")
            raise

    def _make_scripts_executable(self, problem_data_dir: Path) -> None:
        """
        Make all quality and metric scripts executable

        Args:
            problem_data_dir: Path to the problem data directory
        """
        # Ensure quality.py is executable
        quality_script = problem_data_dir / "quality.py"
        if quality_script.exists():
            os.chmod(quality_script, 0o755)
            logger.debug(f"Made {quality_script} executable")
        else:
            logger.warning(f"quality.py script not found at {quality_script}")
            # Check if there's a quality script without .py extension
            alt_quality_script = problem_data_dir / "quality"
            if alt_quality_script.exists():
                os.chmod(alt_quality_script, 0o755)
                logger.debug(f"Made {alt_quality_script} executable")
            else:
                logger.warning(f"quality script not found at {alt_quality_script} either")

        # Make all metric scripts executable
        metrics_dir = problem_data_dir / "metrics"
        if metrics_dir.exists() and metrics_dir.is_dir():
            for script in metrics_dir.glob("**/*"):
                if script.is_file() and not script.name.startswith("."):
                    os.chmod(script, 0o755)
                    logger.debug(f"Made metric script {script.name} executable")

        # Also check for quality/scripts directory in v5
        quality_scripts_dir = problem_data_dir / "quality" / "scripts"
        if quality_scripts_dir.exists() and quality_scripts_dir.is_dir():
            for script in quality_scripts_dir.glob("**/*"):
                if script.is_file() and not script.name.startswith("."):
                    os.chmod(script, 0o755)
                    logger.debug(f"Made quality script {script.name} executable")

        # Look for any .sh scripts and make them executable too
        for script in problem_data_dir.glob("**/*.sh"):
            if script.is_file():
                os.chmod(script, 0o755)
                logger.debug(f"Made shell script {script.name} executable")

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """
        Score the answer to the problem using the AIQ v5.0 protocol quality.py script.

        Args:
            problem: The problem being scored
            agent_workdir: Path to the agent's work directory
            agent_answer_dir: Path to the agent's answer directory
            container_name: Name of the Docker container

        Returns:
            Tuple of (score, error_message, discussion)
        """
        if not isinstance(problem, AIQProjectProblem):
            logger.error("Not an AIQ project problem")
            return 0.0, "Not an AIQ project problem", None

        try:
            # Read initial quality metrics
            initial_quality_path = Path(agent_workdir) / "initial_quality.json"
            logger.info(f"Looking for initial quality file at {initial_quality_path}")

            if not initial_quality_path.exists():
                logger.error(f"Initial quality baseline not found at {initial_quality_path}")
                return 0.0, "Initial quality baseline not found", None

            try:
                with open(initial_quality_path, "r") as f:
                    initial_quality_str = f.read()
                    logger.debug(f"Initial quality content preview: {initial_quality_str[:200]}...")
                    initial_quality = json.loads(initial_quality_str)
                    logger.info(f"Successfully loaded initial quality data with keys: {list(initial_quality.keys())}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse initial quality JSON: {e}")
                return 0.0, f"Invalid initial quality data: {e}", None

            # Check if there's an initial aggregate file available
            initial_aggregate_path = Path(agent_workdir) / "initial_aggregate.json"
            initial_bounded = None
            initial_unbounded = None

            if initial_aggregate_path.exists():
                try:
                    logger.info(f"Found initial aggregate file at {initial_aggregate_path}")
                    with open(initial_aggregate_path, "r") as f:
                        initial_aggregate_data = json.loads(f.read().strip())
                        logger.debug(f"Loaded initial aggregate data: {initial_aggregate_data}")
                        initial_bounded = initial_aggregate_data.get("aggregate_bounded")
                        initial_unbounded = initial_aggregate_data.get("aggregate_unbounded_index")
                        logger.debug(f"Found initial aggregates - bounded: {initial_bounded}, unbounded: {initial_unbounded}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    logger.warning(f"Failed to read initial aggregate data: {e}")

            # If we couldn't get aggregates from initial_aggregate.json, try to extract from initial_quality.json
            if initial_bounded is None:
                initial_bounded = initial_quality.get("aggregate_bounded")
            if initial_unbounded is None:
                initial_unbounded = initial_quality.get("aggregate_unbounded_index")

            # Fall back to v4 fields if v5 fields are not found
            if initial_bounded is None:
                initial_bounded = initial_quality.get("aggregate")
                logger.debug(f"Falling back to v4 'aggregate' field for initial bounded: {initial_bounded}")

            # Run final quality check using the v5 CLI --aggregate-only option for clean output
            logger.debug(f"Running final quality check with --aggregate-only in container {container_name}")

            # Use helper to run command and handle fallbacks
            agg_success, agg_stdout, agg_stderr = await self._run_quality_command(
                container_name=container_name,
                command_type="aggregate",
                output_file="/tmp/final_aggregate.json"
            )

            if not agg_success:
                logger.error(f"Final quality aggregate check failed: {agg_stderr}")
                logger.error(f"Stdout: {agg_stdout}")
            else:
                # Copy aggregate output from container
                subprocess.run(
                    ["docker", "cp",
                     f"{container_name}:/tmp/final_aggregate.json",
                     f"{agent_answer_dir}/final_aggregate.json"],
                    check=False
                )

            # Now run the full quality check to get detailed metrics
            logger.info(f"Running full quality check in container {container_name}")

            # Use helper to run command and handle fallbacks
            result_success, result_stdout, result_stderr = await self._run_quality_command(
                container_name=container_name,
                command_type="standard",
                output_file="/tmp/final_quality.json"
            )

            if not result_success:
                logger.error(f"Final quality check failed: {result_stderr}")
                logger.error(f"Stdout: {result_stdout}")
                if not agg_success:
                    # If both checks failed, we can't calculate a score
                    return 0.0, f"Final quality check failed: {result_stderr}", None
            else:
                # Copy final quality output from container to answer directory
                logger.debug(f"Copying final quality output from container to {agent_answer_dir}/final_quality.json")
                cp_result = subprocess.run(
                    ["docker", "cp",
                     f"{container_name}:/tmp/final_quality.json",
                     f"{agent_answer_dir}/final_quality.json"],
                    capture_output=True,
                    text=True,
                    check=False
                )

                if cp_result.returncode != 0:
                    logger.error(f"Failed to copy final quality file: {cp_result.stderr}")

            # Also run with --diagnose flag to get more detailed diagnostics
            logger.info(f"Running quality check with diagnostics in container {container_name}")

            # Use helper to run command and handle fallbacks
            diag_success, diag_stdout, diag_stderr = await self._run_quality_command(
                container_name=container_name,
                command_type="diagnose",
                output_file="/tmp/final_quality_diag.json"
            )

            if diag_success:
                # Copy diagnostic output from container
                subprocess.run(
                    ["docker", "cp",
                     f"{container_name}:/tmp/final_quality_diag.json",
                     f"{agent_answer_dir}/final_quality_diag.json"],
                    check=False
                )
            else:
                logger.warning(f"Diagnostic quality check failed: {diag_stderr}")

            # Read final quality metrics - prioritize aggregate-only output if available
            final_quality = None
            aggregate_bounded = None
            aggregate_unbounded = None

            # First try to read from aggregate-only output
            aggregate_path = f"{agent_answer_dir}/final_aggregate.json"
            if os.path.exists(aggregate_path):
                try:
                    logger.debug(f"Reading aggregate data from {aggregate_path}")
                    with open(aggregate_path, "r") as f:
                        aggregate_data = json.loads(f.read().strip())
                        logger.debug(f"Loaded aggregate data: {aggregate_data}")
                        aggregate_bounded = aggregate_data.get("aggregate_bounded")
                        aggregate_unbounded = aggregate_data.get("aggregate_unbounded_index")
                        logger.debug(f"Found aggregates - bounded: {aggregate_bounded}, unbounded: {aggregate_unbounded}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    logger.error(f"Failed to read aggregate data: {e}")

            # Then try to read full quality data
            try:
                final_quality_path = f"{agent_answer_dir}/final_quality.json"
                logger.debug(f"Reading final quality data from {final_quality_path}")

                with open(final_quality_path, "r") as f:
                    final_quality_str = f.read()
                    logger.debug(f"Final quality content preview: {final_quality_str[:200]}...")
                    final_quality = json.loads(final_quality_str)
                    logger.debug(f"Successfully loaded final quality data with keys: {list(final_quality.keys())}")

                    # If we couldn't get aggregates from aggregate-only output, get them from full output
                    if aggregate_bounded is None:
                        aggregate_bounded = final_quality.get("aggregate_bounded")
                    if aggregate_unbounded is None:
                        aggregate_unbounded = final_quality.get("aggregate_unbounded_index")

                    # Fall back to v4 fields if v5 fields are not found in final quality data
                    if aggregate_bounded is None:
                        aggregate_bounded = final_quality.get("aggregate")
                        logger.debug(f"Falling back to v4 'aggregate' field for final bounded: {aggregate_bounded}")

            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Failed to read final quality data: {e}")
                if aggregate_bounded is None and aggregate_unbounded is None:
                    return 0.0, f"Failed to read final quality data: {e}", None

            # If we still don't have final_quality data but we have aggregates, create a minimal data structure
            if final_quality is None:
                logger.warning("Could not load detailed quality data, creating minimal structure from aggregates")
                final_quality = {
                    "version": initial_quality.get("version", "5.0"),
                    "era": initial_quality.get("era", "0"),
                    "aggregate_bounded": aggregate_bounded,
                    "aggregate_unbounded_index": aggregate_unbounded,
                    "metrics": []
                }

            # Try to load diagnostic data if available
            diagnostics = None
            diag_path = f"{agent_answer_dir}/final_quality_diag.json"
            if os.path.exists(diag_path):
                try:
                    with open(diag_path, "r") as f:
                        diag_data = json.loads(f.read())
                        diagnostics = diag_data.get("diagnostics")
                        logger.debug(f"Loaded diagnostics: {diagnostics}")
                except (json.JSONDecodeError, FileNotFoundError) as e:
                    logger.error(f"Failed to read diagnostic data: {e}")

            # Extract aggregate metrics for scoring
            # v5 uses aggregate_bounded and aggregate_unbounded_index
            final_bounded = aggregate_bounded
            final_unbounded = aggregate_unbounded

            # Log the aggregate values found
            logger.info(f"Initial aggregate values - bounded: {initial_bounded}, unbounded: {initial_unbounded}")
            logger.info(f"Final aggregate values - bounded: {final_bounded}, unbounded: {final_unbounded}")

            # Build discussion with detailed context
            discussion = []
            discussion.append(f"Project: {problem.problem_id}")
            discussion.append(f"AIQ Protocol Version: {final_quality.get('version', 'unknown')}")
            discussion.append(f"Era: {final_quality.get('era', 'unknown')}")
            discussion.append("")

            # Add bounded metrics if available
            if initial_bounded is not None and final_bounded is not None:
                bounded_improvement = final_bounded - initial_bounded
                logger.info(f"Bounded improvement: {bounded_improvement:+.2f} (from {initial_bounded:.2f} to {final_bounded:.2f})")

                discussion.append(f"Bounded Metrics:")
                discussion.append(f"  Initial: {initial_bounded:.2f}")
                discussion.append(f"  Final:   {final_bounded:.2f}")
                discussion.append(f"  Change:  {bounded_improvement:+.2f}")
                discussion.append("")
            else:
                logger.warning(f"One or both bounded metrics are None - initial: {initial_bounded}, final: {final_bounded}")
                discussion.append(f"Bounded Metrics: Not available")
                discussion.append("")

            # Add unbounded metrics if available
            if initial_unbounded is not None and final_unbounded is not None:
                unbounded_improvement = final_unbounded - initial_unbounded
                pct_change = (unbounded_improvement / max(1, initial_unbounded)) * 100
                logger.info(f"Unbounded improvement: {unbounded_improvement:+.2f} ({pct_change:+.2f}%) (from {initial_unbounded:.2f} to {final_unbounded:.2f})")

                discussion.append(f"Unbounded Metrics:")
                discussion.append(f"  Initial: {initial_unbounded:.2f}")
                discussion.append(f"  Final:   {final_unbounded:.2f}")
                discussion.append(f"  Change:  {unbounded_improvement:+.2f} ({pct_change:+.2f}%)")
                discussion.append("")
            else:
                logger.warning(f"One or both unbounded metrics are None - initial: {initial_unbounded}, final: {final_unbounded}")
                discussion.append(f"Unbounded Metrics: Not available")
                discussion.append("")

            # Include individual metrics if available
            if final_quality.get("metrics"):
                discussion.append("Individual Metrics:")
                for metric in final_quality.get("metrics", []):
                    metric_id = metric.get("id", "unknown")
                    metric_name = metric.get("name", "Unknown Metric")
                    metric_score = metric.get("score")
                    is_bounded = metric.get("bounded", False)
                    if metric_score is not None:
                        discussion.append(f"  {metric_id}: {metric_name} = {metric_score}")
                        logger.debug(f"Metric {metric_id} ({metric_name}): {metric_score}")

            # Include diagnostic information if available
            if diagnostics:
                # Add unbounded delta information from diagnostics
                if diagnostics.get("unbounded_deltas"):
                    discussion.append("")
                    discussion.append("Unbounded Metric Changes:")
                    for delta in diagnostics.get("unbounded_deltas", []):
                        delta_id = delta.get("id", "unknown")
                        current = delta.get("current_score", "N/A")
                        baseline = delta.get("baseline_score", "N/A")
                        change_pct = delta.get("change_pct", "N/A")
                        if change_pct not in (None, "N/A"):
                            direction = "improved" if change_pct > 0 else "regressed"
                            discussion.append(f"  {delta_id}: {current} (baseline: {baseline}, {direction} by {abs(change_pct):.2f}%)")

                # Add information about stalled and failing metrics
                if diagnostics.get("stalled"):
                    stalled_metrics = ", ".join(diagnostics["stalled"])
                    logger.info(f"Stalled metrics: {stalled_metrics}")
                    discussion.append("")
                    discussion.append(f"Stalled metrics (score=0): {stalled_metrics}")

                if diagnostics.get("failing"):
                    failing_metrics = ", ".join(diagnostics["failing"])
                    logger.info(f"Failing metrics: {failing_metrics}")
                    discussion.append("")
                    discussion.append(f"Failing metrics (score=null): {failing_metrics}")

            # Log metric shift if a new era was started
            metric_shift = final_quality.get("metric_shift")
            if metric_shift:
                logger.info(f"Metric shift detected: {metric_shift}")
                discussion.append("")
                discussion.append(f"Metric Shift: {metric_shift}")

            # Calculate score - use the bounded aggregate if available, otherwise unbounded
            # We normalize to 0-1 range for the benchmark framework
            logger.debug("Calculating final score")
            score = 0.0

            # Primary scoring:
            # 1. First try normalized bounded score (0-100 to 0-1)
            # 2. Then try improvement ratio from unbounded index
            # 3. Final fallback is checking for improvement and assigning small score
            if final_bounded is not None:
                # Normalize bounded score (0-100 range) to 0-1
                score = final_bounded / 100.0
                logger.debug(f"Using bounded score: {final_bounded} / 100.0 = {score}")
            elif final_unbounded is not None and initial_unbounded is not None and initial_unbounded > 0:
                # For unbounded, we use improvement ratio relative to baseline
                # Score of 1.0 if doubled or more, 0.0 if no change or worse
                improvement_ratio = (final_unbounded / initial_unbounded) - 1.0
                # Map improvement_ratio to 0.0-1.0 range (100% improvement = 1.0 score)
                score = min(1.0, max(0.0, improvement_ratio))
                logger.debug(f"Using unbounded improvement ratio: {improvement_ratio} (capped to 0.0-1.0) = {score}")

            # Special case: if score would be very close to 0.0 but there's clearly improvement
            if score < 0.01:
                logger.warning("Score is very close to 0.0 - checking for specific improvement indicators")

                # Check if there was any improvement in bounded metrics
                if final_bounded is not None and initial_bounded is not None and final_bounded > initial_bounded:
                    improvement = min(0.05, (final_bounded - initial_bounded) / 100.0)
                    logger.debug(f"Using small score from bounded improvement: {improvement}")
                    score = improvement
                # Check if there was any improvement in unbounded metrics
                elif final_unbounded is not None and initial_unbounded is not None and final_unbounded > initial_unbounded:
                    # Ensure a minimum score for any positive change
                    improvement = min(0.05, (final_unbounded - initial_unbounded) / initial_unbounded)
                    logger.debug(f"Using small score from unbounded improvement: {improvement}")
                    score = improvement

                # Also check individual metrics if overall aggregates don't show improvement
                elif diagnostics and diagnostics.get("unbounded_deltas"):
                    # Check if any individual metrics improved
                    improved_metrics = []
                    for delta in diagnostics.get("unbounded_deltas", []):
                        change_pct = delta.get("change_pct", 0)
                        if change_pct is not None and change_pct > 0:
                            improved_metrics.append(delta.get("id", "unknown"))

                    if improved_metrics:
                        improvement = min(0.05, len(improved_metrics) * 0.01)
                        logger.debug(f"Found {len(improved_metrics)} improved metrics: {', '.join(improved_metrics)}")
                        logger.debug(f"Using small score from individual metric improvements: {improvement}")
                        score = improvement

            logger.info(f"Final score: {score}")
            return score, None, "\n".join(discussion)

        except Exception as e:
            logger.exception(f"Error scoring AIQ problem: {e}")
            return 0.0, f"Scoring error: {str(e)}", None

    async def _run_command(self, *args) -> Tuple[bool, str, str]:
        """Run a command and return success, stdout, and stderr"""
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await proc.communicate()
            return proc.returncode == 0, stdout.decode(), stderr.decode()

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return False, "", str(e)

    async def _run_quality_command(
        self,
        container_name: str,
        command_type: str,
        output_file: str
    ) -> Tuple[bool, str, str]:
        """
        Run a quality command with multiple fallbacks to handle different quality script implementations.

        Args:
            container_name: Name of the Docker container
            command_type: Type of command ('standard', 'aggregate', or 'diagnose')
            output_file: Path to save the output

        Returns:
            Tuple of (success, stdout, stderr)
        """
        command_args = ""
        if command_type == "aggregate":
            command_args = "--aggregate-only"
        elif command_type == "diagnose":
            command_args = "--diagnose"
        command_args += " --no-instructions"

        # Try Python-based quality.py first (AIQ v5)
        cmd = f"cd /home/agent/workdir && python quality.py {command_args} > {output_file}"
        logger.debug(f"Trying Python-based quality command: {cmd}")
        success, stdout, stderr = await self._run_command(
            "docker", "exec", container_name, "bash", "-c", cmd
        )

        return success, stdout, stderr


class LinalgAIQBenchmark(BaseProjectAIQBenchmark):
    """
    Specialized benchmark for the linear algebra project in the AIQ protocol.

    This benchmark creates multiple instances of the same linear algebra project,
    allowing for more reliable performance evaluation and reduced variance.
    """

    name = "linalg_aiq"
    project_name = "linalg"


class CSVParsingAIQBenchmark(BaseProjectAIQBenchmark):
    """
    Specialized benchmark for the CSV parsing project in the AIQ protocol.

    This benchmark creates multiple instances of the same CSV parsing project,
    allowing for more reliable performance evaluation and reduced variance.
    """

    name = "csv_parsing_aiq"
    project_name = "csv_parsing"


class MessagingAppAIQBenchmark(BaseProjectAIQBenchmark):
    """
    Specialized benchmark for the messaging app project in the AIQ protocol.

    This benchmark creates multiple instances of the same messaging app project,
    allowing for more reliable performance evaluation and reduced variance.
    """

    name = "messaging_app_aiq"
    project_name = "messaging_app"


class DistKVStoreAIQBenchmark(BaseProjectAIQBenchmark):
    """
    Specialized benchmark for the distributed key-value store project in the AIQ protocol.

    This benchmark creates multiple instances of the same distributed key-value store project,
    allowing for more reliable performance evaluation and reduced variance.
    """

    name = "dist_kv_store_aiq"
    project_name = "dist_kv_store"
