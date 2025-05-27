# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Some parsing utilities.

This module provides utilities for parsing various types of data,
particularly focusing on numerical parsing from strings.
"""

import re

from typing import Optional, Literal

def extract_before_last(text: str, pattern: str, keep_pattern: bool = False) -> str:
    last_pos = text.rfind(pattern)
    offset = len(pattern) if keep_pattern else 0
    return text[:last_pos + offset] if last_pos != -1 else ""

def extract_after_last(text: str, pattern: str, keep_pattern: bool = False) -> str:
    last_pos = text.rfind(pattern)
    offset = 0 if keep_pattern else len(pattern)
    return text[last_pos + offset:] if last_pos != -1 else ""


def extract_after_first(text: str, pattern: str, keep_pattern: bool = False) -> str:
    first_pos = text.find(pattern)
    offset = 0 if keep_pattern else len(pattern)
    return text[first_pos + offset:] if first_pos != -1 else ""


def extract_between_patterns(
    s: str,
    pattern_a: str,
    pattern_b: str,
    a_occurrence: Literal["first"] | Literal["last"] = "first",
    b_occurrence: Literal["first"] | Literal["last"] = "last",
) -> str | None:
    # Validate both occurrences upfront
    if a_occurrence not in ("first", "last"):
        raise ValueError("Invalid value for a_occurrence. Use 'first' or 'last'.")
    if b_occurrence not in ("first", "last"):
        raise ValueError("Invalid value for b_occurrence. Use 'first' or 'last'.")

    # Determine the index for `pattern_a`
    if a_occurrence == "first":
        start_index = s.find(pattern_a)
    else:  # "last"
        start_index = s.rfind(pattern_a)

    if start_index == -1:
        return None

    start_index += len(pattern_a)

    # Determine the index for `pattern_b`
    if b_occurrence == "first":
        end_index = s.find(pattern_b)
    else:  # "last"
        end_index = s.rfind(pattern_b)

    if end_index == -1 or end_index <= start_index:
        return None

    return s[start_index:end_index]


def parse_number_from_string(
    answer: str,
) -> tuple[bool, Optional[float], Optional[str]]:
    cleaned = answer.strip().replace(",", "")

    # Pattern for a valid number segment
    number_pattern = r"-?\d*\.?\d+(?:[eE][-+]?\d+)?"
    match = re.search(number_pattern, cleaned)

    if not match:
        return (
            False,
            None,
            "Could not find a number in the answer. Please provide a clear numerical response.",
        )

    matched_str = match.group()
    # Check for multiple decimal points in the matched string
    if matched_str.count(".") > 1:
        return (
            False,
            None,
            "Found what looks like a number but couldn't parse it: too many decimal points",
        )

    try:
        value = float(matched_str)
        full_match = matched_str == cleaned
        if not full_match:
            return (
                True,
                value,
                "Warning: Found additional text around the number. In future, try to provide just the number.",
            )
        return True, value, None
    except ValueError as e:
        return (
            False,
            None,
            f"Found what looks like a number but couldn't parse it: {str(e)}",
        )
