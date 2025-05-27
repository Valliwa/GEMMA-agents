# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the Calculator tool."""
import pytest
import time
import statistics
from typing import List, Dict, Any, Callable, Awaitable

from src.tools.calculator import Calculator
from src.types.tool_types import ToolResult
from src.agents.implementations import DemoAgent


# Apply asyncio marker only to the specific tests that need it
@pytest.mark.asyncio
async def test_valid_expressions(expression, expected_result):
    """Test calculator with valid expressions."""
    # Use the project's DemoAgent instead of a custom mock
    calculator = Calculator(
        calling_agent=DemoAgent(),
        reasoning="Testing calculator",
        expression=expression
    )
    
    result = await calculator.run()
    
    assert result.success is True
    assert result.tool_name == Calculator.TOOL_NAME
    assert result.output == expected_result


# Use pytest.mark.parametrize outside of the function definition
test_valid_expressions = pytest.mark.parametrize("expression, expected_result", [
    ("2 + 2", "4"),  # Basic addition
    ("10 - 5", "5"),  # Basic subtraction
    ("3 * 4", "12"),  # Multiplication
    ("8 / 4", "2.0"),  # Division
    ("2 + 3 * 4", "14"),  # Order of operations
    ("(2 + 3) * 4", "20"),  # Parentheses
    ("2 + 2.5", "4.5"),  # Floating point
    ("10 / 3", "3.3333333333333335"),  # Division with floating point result
])(test_valid_expressions)


@pytest.mark.asyncio
async def test_invalid_expressions(expression):
    """Test calculator with invalid expressions that should raise errors."""
    calculator = Calculator(
        calling_agent=DemoAgent(),
        reasoning="Testing invalid expressions",
        expression=expression
    )
    
    result = await calculator.run()
    
    assert result.success is False
    assert result.tool_name == Calculator.TOOL_NAME
    assert result.errors is not None and result.errors != ""


# Use pytest.mark.parametrize outside of the function definition
test_invalid_expressions = pytest.mark.parametrize("expression", [
    "1 / 0",  # Division by zero
    "1 + ",  # Incomplete expression
    "* 2",  # Missing operand
    "(1 + 2",  # Unclosed parenthesis
])(test_invalid_expressions)


@pytest.mark.asyncio
async def test_examples_are_valid():
    """Test that the examples provided by the tool actually work."""
    examples = Calculator.generate_examples()
    
    # Test each example from the documentation
    for tool_instance, expected_result in examples:
        # Run the calculator
        result = await tool_instance.run()
        
        # Verify the result matches the expected result
        assert result.success == expected_result.success
        
        # Handle the case where numeric values might be returned with decimal points
        if result.output and expected_result.output:
            try:
                # Try to convert both to floats for numeric comparison
                result_float = float(result.output)
                expected_float = float(expected_result.output)
                assert result_float == expected_float
            except ValueError:
                # If conversion fails, fall back to string comparison
                assert result.output == expected_result.output
        else:
            assert result.output == expected_result.output
            
        assert result.tool_name == expected_result.tool_name


# This function is not async, so we don't apply the asyncio marker to it
def test_tool_metadata():
    """Test that the tool has the expected metadata."""
    # Check class-level attributes
    assert Calculator.TOOL_NAME == "calculate"
    assert "calculator" in Calculator.TOOL_DESCRIPTION.lower()
    
    # Check required class methods
    assert hasattr(Calculator, "generate_examples")
    examples = Calculator.generate_examples()
    assert isinstance(examples, list)
    assert len(examples) > 0


# ---------------------- Performance Testing ----------------------

async def benchmark_async_function(func: Callable[..., Awaitable[Any]], 
                                  iterations: int = 5) -> Dict[str, Any]:
    """Benchmark an async function's performance."""
    durations = []
    
    for _ in range(iterations):
        start_time = time.perf_counter()
        await func()
        end_time = time.perf_counter()
        durations.append(end_time - start_time)
    
    return {
        'mean': statistics.mean(durations),
        'median': statistics.median(durations),
        'stdev': statistics.stdev(durations) if len(durations) > 1 else 0,
        'min': min(durations),
        'max': max(durations),
        'iterations': iterations
    }


@pytest.mark.asyncio
@pytest.mark.performance
async def test_calculator_simple_performance():
    """Test the performance of the calculator with simple expressions."""
    agent = DemoAgent()
    
    async def run_calculator():
        calculator = Calculator(
            calling_agent=agent,
            reasoning="Performance testing simple expression",
            expression="1 + 1"
        )
        return await calculator.run()
    
    # Run the benchmark
    results = await benchmark_async_function(run_calculator, iterations=50)
    
    # Log performance data for analysis
    print(f"\nCalculator Performance (Simple Expression):")
    print(f"  Mean: {results['mean']:.6f}s")
    print(f"  Median: {results['median']:.6f}s")
    print(f"  StdDev: {results['stdev']:.6f}s")
    print(f"  Min: {results['min']:.6f}s")
    print(f"  Max: {results['max']:.6f}s")
    
    # Establish performance baseline (adjust based on observed performance)
    # This is a very simple operation, so it should be fast
    assert results['median'] < 0.1, "Simple calculator operation performance is below threshold"


@pytest.mark.asyncio
@pytest.mark.performance
async def test_calculator_complex_performance():
    """Test the performance of the calculator with complex expressions."""
    agent = DemoAgent()
    
    async def run_calculator():
        calculator = Calculator(
            calling_agent=agent,
            reasoning="Performance testing complex expression",
            expression="(2 + 3) * (4 + 5) / (6 + 7) ** 2 + 8 - 9"
        )
        return await calculator.run()
    
    # Run the benchmark
    results = await benchmark_async_function(run_calculator, iterations=20)
    
    # Log performance data for analysis
    print(f"\nCalculator Performance (Complex Expression):")
    print(f"  Mean: {results['mean']:.6f}s")
    print(f"  Median: {results['median']:.6f}s")
    print(f"  StdDev: {results['stdev']:.6f}s")
    print(f"  Min: {results['min']:.6f}s")
    print(f"  Max: {results['max']:.6f}s")
    
    # Establish performance baseline (adjust based on observed performance)
    # Complex operations should still be relatively fast
    assert results['median'] < 0.15, "Complex calculator operation performance is below threshold"


@pytest.mark.asyncio
@pytest.mark.performance
@pytest.mark.parametrize("expression", [
    "1 + 2",  # Simple
    "1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10",  # Medium (more terms)
    "((1 + 2) * 3 - 4) / 5 + ((6 - 7) * 8 / 9) ** 2"  # Complex (nested operations)
])
async def test_calculator_parametrized_performance(expression):
    """Test calculator performance with various expression complexities."""
    agent = DemoAgent()
    
    async def run_calculator():
        calculator = Calculator(
            calling_agent=agent,
            reasoning=f"Performance testing expression: {expression}",
            expression=expression
        )
        return await calculator.run()
    
    # Run fewer iterations for parametrized tests
    results = await benchmark_async_function(run_calculator, iterations=10)
    
    # Log performance data
    complexity = "Simple" if len(expression) < 10 else "Medium" if len(expression) < 30 else "Complex"
    print(f"\nCalculator Performance ({complexity}):")
    print(f"  Expression: {expression}")
    print(f"  Median: {results['median']:.6f}s")
    
    # Dynamic threshold based on complexity
    threshold = 0.05 if complexity == "Simple" else 0.1 if complexity == "Medium" else 0.2
    assert results['median'] < threshold, f"{complexity} calculator operation performance is below threshold"
