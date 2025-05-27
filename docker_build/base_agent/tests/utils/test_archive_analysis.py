# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the archive analysis utility."""
import os
import json
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from src.utils.archive_analysis import (
    ArchiveAnalyzer,
    compute_statistics,
    ScoreType,
)

@pytest.fixture
def test_archive(tmp_path):
    """Create a minimal test archive with realistic data."""
    archive_root = tmp_path / "results" / "run_1"
    archive_root.mkdir(parents=True)

    # Create two agent iterations with two benchmarks each
    for agent_i in range(2):  # agent_0 and agent_1
        agent_dir = archive_root / f"agent_{agent_i}"
        agent_dir.mkdir()

        # Add description
        desc_file = agent_dir / "agent_code" / "description.txt"
        desc_file.parent.mkdir(parents=True)
        desc_file.write_text(f"Test agent version {agent_i}")

        # Create metadata.json
        metadata = {
            "experiment_start_timestamp": "2025-03-15T11:24:37.912936",
            "python_version": "3.12.9 | packaged by conda-forge | (main, Mar 4 2025, 22:48:41) [GCC 13.3.0]",
            "executable": "/usr/bin/python",
            "agent_iteration": agent_i,
            "git_commit": "21c6c657b2835445e8824c04b7527a655a215751",
            "current_benchmark_idx": -1
        }
        with open(agent_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        for bench_name in ["gsm8k", "math"]:
            bench_dir = agent_dir / "benchmarks" / bench_name
            bench_dir.mkdir(parents=True)

            # Create realistic results.jsonl based on provided example
            results = [
                {
                    "problem_id": f"{bench_name}_{i}",
                    "timestamp": (datetime.now() + timedelta(seconds=i)).isoformat(),
                    "score": float(i % 2),  # Alternate between 0.0 and 1.0
                    "tokens_used": 57697 + i * 1000,
                    "num_cached_tokens": i * 500,
                    "cost_estimate": 0.05297255 + i * 0.01,
                    "wall_time": 120.31996870040894 - i * 10,  # Vary the times
                    "timed_out": i == 3,  # Make one problem timeout
                    "cost_threshold_exceeded": False
                }
                for i in range(4)  # 4 problems per benchmark
            ]

            with open(bench_dir / "results.jsonl", "w") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")

            # Create corresponding perf.json based on provided example
            perf_data = {
                "avg_score": sum(r["score"] for r in results) / len(results),
                "tokens": sum(r["tokens_used"] for r in results),
                "num_cached_tokens": sum(r["num_cached_tokens"] for r in results),
                "cost": sum(r["cost_estimate"] for r in results),
                "time": sum(r["wall_time"] for r in results)
            }

            with open(bench_dir / "perf.json", "w") as f:
                json.dump(perf_data, f)

            # Create trace directory with summaries
            for result in results:
                prob_dir = bench_dir / "traces" / result["problem_id"]
                prob_dir.mkdir(parents=True)

                # Create a meaningful summary based on performance
                if result["score"] > 0:
                    summary = (
                        f"Analysis for problem {result['problem_id']}\n"
                        f"Score: {result['score']}\n"
                        "The agent successfully solved this problem using an efficient approach.\n"
                        f"Used {result['tokens_used']} tokens ({result['num_cached_tokens']} cached)\n"
                        f"Cost: ${result['cost_estimate']:.4f}\n"
                        f"Time: {result['wall_time']:.2f}s\n"
                        "Key steps: problem decomposition, accurate calculation, clear answer format."
                    )
                else:
                    summary = (
                        f"Analysis for problem {result['problem_id']}\n"
                        f"Score: {result['score']}\n"
                        "The agent failed to solve this problem due to:\n"
                        "- Incorrect problem interpretation\n"
                        "- Calculation errors\n"
                        f"Used {result['tokens_used']} tokens ({result['num_cached_tokens']} cached)\n"
                        f"Cost: ${result['cost_estimate']:.4f}\n"
                        f"Time: {result['wall_time']:.2f}s"
                    )
                    if result["timed_out"]:
                        summary += "\nExecution timed out before completion."

                (prob_dir / "summary.txt").write_text(summary)
                (prob_dir / "execution_tree.txt").write_text(
                    "Mock execution tree for test purposes...\n"
                    "agent_call -> tool_call -> result"
                )

    return archive_root

def test_core_statistics(test_archive):
    """Test basic statistical analysis of archive runs."""
    analyzer = ArchiveAnalyzer(test_archive)
    df = analyzer.analyze_run()

    # Check basic structure
    assert len(df) == 2  # Two iterations
    assert "gsm8k_score" in df.columns
    assert "math_score" in df.columns

    # Verify metric calculations
    assert all(df["avg_tokens"] > 0)
    assert all(df["avg_cost"] > 0)
    assert all(df["avg_time"] > 0)
    assert all(df["utility_score"] >= 0)

    # Check specific values based on our synthetic data
    # Each benchmark has 4 problems, alternating scores of 0 and 1
    assert abs(df.loc[0, "gsm8k_score"] - 0.5) < 1e-6  # Should be 0.5
    assert abs(df.loc[0, "math_score"] - 0.5) < 1e-6  # Should be 0.5

def test_problem_analysis(test_archive):
    """Test problem-level analysis functions."""
    analyzer = ArchiveAnalyzer(test_archive)

    # Test worst problems analysis
    worst = analyzer.get_worst_performing_problems(
        iteration=0,
        n=2,
        benchmark="gsm8k"
    )

    assert len(worst) == 2
    assert all(row["score"] == 0.0 for _, row in worst.iterrows())
    assert all("failed to solve" in row["summary"] for _, row in worst.iterrows())

    # Test best problems analysis
    best = analyzer.get_best_performing_problems(
        iteration=0,
        n=2,
        benchmark="gsm8k"
    )

    assert len(best) == 2
    assert all(row["score"] == 1.0 for _, row in best.iterrows())
    assert all("successfully solved" in row["summary"] for _, row in best.iterrows())

def test_trend_analysis(test_archive):
    """Test trend analysis functions."""
    analyzer = ArchiveAnalyzer(test_archive)
    trend_df = analyzer.get_improvement_trend()

    assert len(trend_df) == 2  # Two iterations
    assert "mean_score" in trend_df.columns
    assert "rolling_avg_3" in trend_df.columns
    assert "rolling_avg_5" in trend_df.columns

    # Check rolling averages
    # With only two points, rolling_avg_3 should equal mean_score
    assert all(trend_df["rolling_avg_3"] == trend_df["mean_score"])

def test_statistical_measures(test_archive):
    """Test statistical measures and confidence intervals."""
    analyzer = ArchiveAnalyzer(test_archive)
    scores_df, summaries_df = analyzer.get_problem_scores_by_iteration()

    stats = compute_statistics(scores_df, summaries_df)

    # Check basic statistics properties
    assert all(stats["ci_lower"] <= stats["target_score"])
    assert all(stats["target_score"] <= stats["ci_upper"])
    assert all(stats["std_score"] >= 0)  # Standard deviation should be non-negative

    # Check that CI width is reasonable (not too wide or narrow)
    ci_width = stats["ci_upper"] - stats["ci_lower"]
    assert all(ci_width >= 0)  # CIs should have positive width
    assert all(ci_width <= 1)  # For 0-1 scores, CI width should be <= 1

def test_edge_cases(test_archive):
    """Test handling of edge cases."""
    analyzer = ArchiveAnalyzer(test_archive)

    # Test non-existent iteration
    with pytest.raises(ValueError):
        analyzer.get_worst_performing_problems(999)

    # Test non-existent benchmark
    with pytest.raises(ValueError):
        analyzer.get_worst_performing_problems(0, benchmark="nonexistent")

    # Test empty directories
    empty_dir = test_archive / "agent_2" / "benchmarks" / "empty"
    empty_dir.mkdir(parents=True)

    # Should handle empty benchmark gracefully
    df = analyzer.analyze_run()
    assert "empty_score" not in df.columns
