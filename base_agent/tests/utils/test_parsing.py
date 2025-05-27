# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for the parsing utilities module.
"""
import re
import pytest
from src.utils.parsing import (
    extract_before_last,
    extract_after_last,
    extract_after_first,
    extract_between_patterns,
    parse_number_from_string,
)


# Test extract_before_last
@pytest.mark.parametrize(
    "text, pattern, keep_pattern, expected",
    [
        ("hello world hello", "hello", False, "hello world "),  # Basic case
        ("hello world hello", "hello", True, "hello world hello"),  # Keep pattern
        ("no pattern here", "xyz", False, ""),  # Pattern not found
        ("", "hello", False, ""),  # Empty string
        ("hello", "hello", False, ""),  # Pattern at end
    ],
    ids=["basic", "keep_pattern", "not_found", "empty", "end_pattern"],
)
def test_extract_before_last(text, pattern, keep_pattern, expected):
    result = extract_before_last(text, pattern, keep_pattern)
    assert result == expected, f"Expected '{expected}', got '{result}'"


# Test extract_after_last
@pytest.mark.parametrize(
    "text, pattern, keep_pattern, expected",
    [
        ("hello world hello", "hello", False, ""),  # Last occurrence at end
        ("hello world hello", "hello", True, "hello"),  # Keep pattern
        ("hello world hello", "world", False, " hello"),  # Middle occurrence
        ("no pattern here", "xyz", False, ""),  # Pattern not found
        ("hello", "hello", True, "hello"),  # Single pattern
    ],
    ids=["end", "keep_pattern", "middle", "not_found", "single"],
)
def test_extract_after_last(text, pattern, keep_pattern, expected):
    result = extract_after_last(text, pattern, keep_pattern)
    assert result == expected


# Test extract_after_first
@pytest.mark.parametrize(
    "text, pattern, keep_pattern, expected",
    [
        ("hello world hello", "hello", False, " world hello"),  # First occurrence
        ("hello world hello", "hello", True, "hello world hello"),  # Keep pattern
        ("no pattern here", "xyz", False, ""),  # Pattern not found
        ("hello", "he", False, "llo"),  # Partial pattern
        ("", "xyz", False, ""),  # Empty string
    ],
    ids=["basic", "keep_pattern", "not_found", "partial", "empty"],
)
def test_extract_after_first(text, pattern, keep_pattern, expected):
    result = extract_after_first(text, pattern, keep_pattern)
    assert result == expected


# Test extract_between_patterns
@pytest.mark.parametrize(
    "text, pattern_a, pattern_b, a_occ, b_occ, expected",
    [
        # First/Last combinations
        ("start middle end", "start", "end", "first", "last", " middle "),
        ("a b a c a d", "a", "a", "first", "last", " b a c "),
        ("a b a c a d", "a", "a", "last", "first", None),  # Invalid range
        # Pattern not found
        ("hello world", "xyz", "abc", "first", "last", None),
        ("hello world", "hello", "xyz", "first", "last", None),
        # Edge cases
        ("", "a", "b", "first", "last", None),  # Empty string
        ("abc", "a", "c", "first", "last", "b"),  # Adjacent patterns
    ],
    ids=[
        "first_last",
        "multiple_a_last",
        "invalid_range",
        "a_missing",
        "b_missing",
        "empty",
        "adjacent",
    ],
)
def test_extract_between_patterns(text, pattern_a, pattern_b, a_occ, b_occ, expected):
    result = extract_between_patterns(text, pattern_a, pattern_b, a_occ, b_occ)
    assert result == expected


# Test extract_between_patterns with invalid occurrence values
@pytest.mark.parametrize(
    "a_occ, b_occ",
    [("invalid", "first"), ("first", "invalid")],
    ids=["invalid_a", "invalid_b"],
)
def test_extract_between_patterns_invalid_occurrence(a_occ, b_occ):
    with pytest.raises(ValueError, match="Invalid value for.*occurrence"):
        extract_between_patterns("text", "a", "b", a_occ, b_occ)


# Fixture for parse_number_from_string tests
@pytest.fixture
def number_parser():
    return parse_number_from_string


# Test parse_number_from_string
@pytest.mark.parametrize(
    "input_str, expected",
    [
        # Successful cases
        ("42", (True, 42.0, None)),
        ("-3.14", (True, -3.14, None)),
        ("1,234.56", (True, 1234.56, None)),  # Commas removed
        ("  6.022e23  ", (True, 6.022e23, None)),  # Scientific notation
        # Success with warning
        ("42 extra text", (True, 42.0, "Warning: Found additional text.*")),
        # Failure cases
        ("no number here", (False, None, "Could not find a number.*")),
        ("", (False, None, "Could not find a number.*")),
    ],
    ids=[
        "integer",
        "negative_float",
        "comma_float",
        "scientific",
        "extra_text",
        "no_number",
        "empty",
    ],
)
def test_parse_number_from_string(number_parser, input_str, expected):
    success, value, message = number_parser(input_str)
    assert success == expected[0]
    assert value == expected[1]
    if message is not None and expected[2] is not None:
        assert re.match(expected[2], message)  # Match regex pattern for message
    else:
        assert message == expected[2]


# Example of a slow test (for demonstration)
@pytest.mark.slow
def test_parse_number_from_string_slow(number_parser):
    import time
    time.sleep(1)  # Simulate slow operation
    success, value, _ = number_parser("12345")
    assert success and value == 12345.0
