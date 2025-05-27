# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the base benchmark components."""
import os
import pytest
import tempfile
import jsonlines
import asyncio
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock

from src.benchmarks.base import BaseBenchmark, Problem, BenchmarkTracker


class SimpleBenchmark(BaseBenchmark):
    """A minimal benchmark implementation for testing."""
    
    name = "simple_benchmark"
    
    def __init__(self, seed=None, subset_size=None):
        super().__init__(seed, subset_size)
        self._all_problems = [
            Problem(
                problem_id=f"problem_{i}",
                statement=f"Test problem {i}",
                answer=f"Answer {i}",
                answer_discussion=f"Discussion {i}"
            )
            for i in range(1, 4)
        ]
    
    @property
    def problems(self) -> list[Problem]:
        if self.subset_size is not None and self.subset_size < len(self._all_problems):
            return self._all_problems[:self.subset_size]
        return self._all_problems
    
    def score_problem(self, problem, agent_workdir, agent_answer_dir, container_name):
        # Simple implementation that returns a score of 0.5 for all problems
        # In a real benchmark, this would evaluate the agent's answer
        return 0.5, None, f"Score for {problem.problem_id}"


class TestProblems:
    """Tests for the Problem class."""
    
    def test_problem_initialization(self):
        """Test that a Problem can be correctly initialized."""
        problem = Problem(
            problem_id="test_id",
            statement="Test statement",
            answer="Test answer",
            answer_discussion="Test discussion"
        )
        
        assert problem.problem_id == "test_id"
        assert problem.statement == "Test statement"
        assert problem.answer == "Test answer"
        assert problem.answer_discussion == "Test discussion"


class TestBenchmarkTracker:
    """Tests for the BenchmarkTracker."""
    
    @pytest.fixture
    def temp_results_file(self):
        """Create a temporary file for benchmark results."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp:
            temp_path = Path(temp.name)
        
        yield temp_path
        
        # Clean up
        if temp_path.exists():
            os.unlink(temp_path)
    
    def test_tracker_initialization(self, temp_results_file):
        """Test that a tracker can be initialized with a new file."""
        tracker = BenchmarkTracker(temp_results_file)
        
        assert isinstance(tracker, BenchmarkTracker)
        assert isinstance(tracker.results, dict)
        assert len(tracker.results) == 0
    
    def test_tracker_with_existing_file(self, temp_results_file):
        """Test tracker initialization with an existing results file."""
        # Create a results file with some data
        with jsonlines.open(temp_results_file, mode="w") as writer:
            writer.write({
                "problem_id": "existing_problem",
                "timestamp": datetime.now().isoformat(),
                "score": 0.75,
                "tokens_used": 1000,
                "num_cached_tokens": 500,
                "cost_estimate": 0.02,
                "wall_time": 10.5,
                "timed_out": False,
                "cost_threshold_exceeded": False
            })
        
        # Initialize a tracker with the existing file
        tracker = BenchmarkTracker(temp_results_file)
        
        assert len(tracker.results) == 1
        assert "existing_problem" in tracker.results
        assert tracker.results["existing_problem"].score == 0.75
    
    def test_start_problem(self, temp_results_file):
        """Test starting a new problem tracking."""
        tracker = BenchmarkTracker(temp_results_file)
        tracker.start_problem("new_problem")
        
        assert "new_problem" in tracker.results
        assert tracker.results["new_problem"].problem_id == "new_problem"
        assert tracker.results["new_problem"].score is None
        assert tracker.results["new_problem"].is_complete() is False
        
        # Verify the result was written to the file
        with jsonlines.open(temp_results_file) as reader:
            results = list(reader)
            assert len(results) == 1
            assert results[0]["problem_id"] == "new_problem"
    
    def test_update_problem(self, temp_results_file):
        """Test updating an existing problem's metrics."""
        tracker = BenchmarkTracker(temp_results_file)
        tracker.start_problem("update_problem")
        
        # Update the problem with complete metrics
        tracker.update_problem(
            problem_id="update_problem",
            tokens_used=1500,
            num_cached_tokens=300,
            cost_estimate=0.03,
            wall_time=15.2,
            score=0.85,
            timed_out=False,
            cost_threshold_exceeded=False
        )
        
        # Check the in-memory state
        assert tracker.results["update_problem"].tokens_used == 1500
        assert tracker.results["update_problem"].score == 0.85
        assert tracker.results["update_problem"].is_complete() is True
        
        # Verify the file was updated
        with jsonlines.open(temp_results_file) as reader:
            results = list(reader)
            assert len(results) == 1
            assert results[0]["problem_id"] == "update_problem"
            assert results[0]["tokens_used"] == 1500
            assert results[0]["score"] == 0.85
    
    def test_update_nonexistent_problem(self, temp_results_file):
        """Test updating a problem that hasn't been started raises an error."""
        tracker = BenchmarkTracker(temp_results_file)
        
        with pytest.raises(KeyError):
            tracker.update_problem("nonexistent", score=1.0)


class TestBaseBenchmark:
    """Tests for the BaseBenchmark class using our simple implementation."""
    
    @pytest.fixture
    def simple_benchmark(self):
        """Create a simple benchmark instance for testing."""
        return SimpleBenchmark()
    
    def test_benchmark_initialization(self, simple_benchmark):
        """Test that a benchmark can be correctly initialized."""
        assert isinstance(simple_benchmark, BaseBenchmark)
        assert simple_benchmark.name == "simple_benchmark"
        assert len(simple_benchmark.problems) == 3
    
    def test_get_problem(self, simple_benchmark):
        """Test retrieving a specific problem by ID."""
        problem = simple_benchmark.get_problem("problem_2")
        
        assert problem is not None
        assert problem.problem_id == "problem_2"
        assert problem.statement == "Test problem 2"
    
    def test_get_nonexistent_problem(self, simple_benchmark):
        """Test that get_problem returns None for nonexistent problems."""
        problem = simple_benchmark.get_problem("nonexistent_problem")
        
        assert problem is None
    
    def test_benchmark_with_subset(self):
        """Test creating a benchmark with a subset of problems."""
        benchmark = SimpleBenchmark(subset_size=2)
        
        assert len(benchmark.problems) == 2
        assert benchmark.problems[0].problem_id == "problem_1"
        assert benchmark.problems[1].problem_id == "problem_2"
    
    def test_score_problem(self, simple_benchmark):
        """Test the scoring mechanism."""
        problem = simple_benchmark.problems[0]
        
        score, errors, discussion = simple_benchmark.score_problem(
            problem, 
            agent_workdir="/fake/workdir",
            agent_answer_dir="/fake/answer_dir",
            container_name="fake_container"
        )
        
        assert score == 0.5
        assert errors is None
        assert discussion == f"Score for {problem.problem_id}"
    
    @pytest.mark.asyncio
    async def test_setup_problem(self, simple_benchmark):
        """Test the default setup_problem implementation."""
        problem = simple_benchmark.problems[0]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # The base implementation is a no-op, but should run without errors
            await simple_benchmark.setup_problem(
                problem,
                problem_data_dir=temp_path,
                container_name="fake_container"
            )
