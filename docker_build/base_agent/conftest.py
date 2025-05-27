# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest

# Enable asyncio support for pytest
pytest_plugins = ["pytest_asyncio"]

# Optional: Define custom command-line options for your markers
def pytest_addoption(parser):
    parser.addoption(
        "--run-llm",
        action="store_true",
        default=False,
        help="Run tests marked with 'uses_llm'",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked with 'slow'",
    )

# Skip tests based on markers unless the corresponding option is provided
def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-llm"):
        skip_llm = pytest.mark.skip(reason="need --run-llm option to run")
        for item in items:
            if "uses_llm" in item.keywords:
                item.add_marker(skip_llm)
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
