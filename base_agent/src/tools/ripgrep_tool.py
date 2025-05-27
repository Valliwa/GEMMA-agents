# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import logging
import subprocess

from pathlib import Path
from pydantic import Field

from .base_tool import BaseTool
from ..types.tool_types import ToolResult
from ..types.agent_types import AgentInterface

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class RipGrepTool(BaseTool):
    """Tool for searching files using ripgrep with context and line numbers"""

    TOOL_NAME = "ripgrep_search"
    TOOL_DESCRIPTION = """Searches for text in files using ripgrep (rg) with configurable options.

Key Features:
- Searches for text strings across multiple files in a directory
- Case-sensitive or case-insensitive search
- Shows context lines around matches
- Displays line numbers in the left gutter
- Limits total number of matches globally to avoid context overflow
- Groups results by file

Usage Tips:
1. Provide a search string and root directory
2. Optional: Set case_sensitive to False for case-insensitive search
3. Optional: Adjust context_lines to show more or less context
4. Optional: Set max_matches to limit the total number of results across all files

If results are limited, the output will indicate how many matches were omitted.
You can search in more specific subdirectories to find additional matches.

Output format matches the file viewer style with <FILE>, <PATH>, <LINES>, and <CONTENT> sections.
"""

    search_string: str = Field(
        ...,
        description="The text string to search for in files",
    )
    root_dir: str = Field(
        ...,
        description="The root directory to search in. This must be a directory and not a file path.",
    )
    case_sensitive: bool = Field(
        default=True,
        description="Whether the search should be case-sensitive",
    )
    context_lines: int = Field(
        default=2,
        description="Number of context lines to show before and after matches",
        ge=0,
        le=5,
    )

    max_matches: int = Field(
        default=20,
        description="Maximum total number of matches to return across all files",
        ge=1,
        le=1000,
    )

    def __init__(self, calling_agent: AgentInterface, **data):
        super().__init__(calling_agent=calling_agent, **data)

    def _parse_ripgrep_output(self, rg_output: bytes) -> tuple[list, int, set[str]]:
        if not rg_output.strip():
            return [], 0, set()

        decoded_output = rg_output.decode("utf-8", errors="replace")
        if decoded_output == "":
            return [], 0, set()

        files = {}
        total_matches = 0
        omitted_files = set()

        def parse_line(line: str) -> tuple[str, int, str, bool] | None:
            match = re.match(r"([^:]+)([:-])(\d+)([:-])(.*)", line)
            if not match:
                return None
            filepath, separator, line_num_str, _, content = match.groups()
            return (filepath, int(line_num_str), content, separator == ":")

        # First pass: Collect all matches and count total
        total_possible_matches = 0
        for line in decoded_output.splitlines():
            if not line.strip() or line.startswith("--"):
                continue

            parsed = parse_line(line)
            if not parsed:
                continue

            filepath, line_num, content, is_match = parsed
            if is_match:
                total_possible_matches += 1

                # If we haven't hit our limit, collect this match
                if total_matches < self.max_matches:
                    if filepath not in files:
                        files[filepath] = {
                            "path": filepath,
                            "matches": [],
                            "match_count": 0
                        }

                    file_info = files[filepath]
                    file_info["matches"].append({
                        "line": line_num,
                        "content": content,
                        "is_match": True
                    })
                    file_info["match_count"] += 1
                    total_matches += 1
                else:
                    omitted_files.add(filepath)

        # Second pass: Add context lines for files that had matches
        for line in decoded_output.splitlines():
            if not line.strip() or line.startswith("--"):
                continue

            parsed = parse_line(line)
            if not parsed:
                continue

            filepath, line_num, content, is_match = parsed
            if not is_match and filepath in files:
                # Add context line to existing file
                file_info = files[filepath]
                # Insert in correct position
                idx = 0
                while (idx < len(file_info["matches"]) and
                       file_info["matches"][idx]["line"] < line_num):
                    idx += 1
                file_info["matches"].insert(idx, {
                    "line": line_num,
                    "content": content,
                    "is_match": False
                })

        # Convert to list format preserving original file order
        result = []
        processed_files = set()

        # Use original output to maintain file order
        for line in decoded_output.splitlines():
            if not line.strip() or line.startswith("--"):
                continue

            filepath = line.split(":")[0].split("-")[0]
            if filepath not in processed_files and filepath in files:
                file_info = files[filepath]
                if file_info["matches"]:
                    result.append({
                        "path": filepath,
                        "matches": sorted(file_info["matches"], key=lambda x: x["line"])
                    })
                    processed_files.add(filepath)

        omitted_count = total_possible_matches - total_matches

        return result, omitted_count, omitted_files

    def _format_viewer_style(self, parsed_results: list, omitted_count: int) -> str:
        """Format results in a style similar to the file viewer."""
        if not parsed_results:
            return "No matches found."

        formatted_output = []

        for file_entry in parsed_results:
            filepath = file_entry["path"]
            matches = file_entry["matches"]

            formatted_output.append(f"<FILE>")
            formatted_output.append(f"<PATH>{filepath}</PATH>")

            if matches:
                max_line = max(match["line"] for match in matches)
                formatted_output.append(f"<LINES>{max_line}</LINES>")
                formatted_output.append("<CONTENT>")
                # Format each line with consistent width line numbers
                line_width = len(str(max_line))
                for match in sorted(matches, key=lambda x: x["line"]):
                    line_num = str(match["line"]).rjust(line_width)
                    # Add a marking for actual matches vs context
                    prefix = " " if not match.get("is_match", True) else ""
                    formatted_output.append(f"{line_num} |{prefix}{match['content']}")

                formatted_output.append("</CONTENT>")
            formatted_output.append("</FILE>")

        if omitted_count > 0:
            formatted_output.append(
                f"\nNote: {omitted_count} additional matches were omitted due to the global limit. "
                "Call the ripgrep tool on specific subdirectories for more results."
            )

        return "\n".join(formatted_output)

    async def run(self) -> ToolResult:
        try:
            root_path = Path(self.root_dir).resolve()
            if not root_path.exists() or not root_path.is_dir():
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=False,
                    errors=f"Root directory {self.root_dir} does not exist or is not a directory",
                )

            # Build ripgrep command with color disabled for simpler parsing
            cmd = ["rg", "--color", "never", "--line-number"]
            if not self.case_sensitive:
                cmd.append("--ignore-case")
            cmd.extend(["--context", str(self.context_lines)])
            # Use a large max-count for ripgrep to get all results, we'll limit in parsing
            cmd.extend(["--max-count", "10000"])
            # cmd.extend(["--glob", "'!agent_outputs/'"])  # exclude the agent outputs from being included
            cmd.append(self.search_string)
            cmd.append(str(root_path))

            try:
                result = subprocess.run(cmd, capture_output=True, check=True)
                logger.debug(f"Raw ripgrep output:\n{result.stdout.decode('utf-8')}")
                parsed_results, omitted_count, omitted_files = self._parse_ripgrep_output(
                    result.stdout
                )
                # Format results in file viewer style
                formatted_output = self._format_viewer_style(
                    parsed_results, omitted_count
                )

                # Create summary
                actual_matches = sum(
                    sum(1 for m in f["matches"] if m.get("is_match", True))
                    for f in parsed_results
                )
                output_dict = {
                    "summary": {
                        "total_matches": actual_matches,
                        "omitted_matches": omitted_count,
                    },
                    "formatted_output": formatted_output,
                }

                # Only include omitted_files if there were any
                if omitted_files:
                    output_dict["omitted_files"] = "\n".join(sorted(omitted_files))

                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=True,
                    output=output_dict,
                )
            except subprocess.CalledProcessError as e:
                if e.returncode == 1:
                    # No matches found (exit code 1 is normal for no matches in ripgrep)
                    return ToolResult(
                        tool_name=self.TOOL_NAME,
                        success=True,
                        output={
                            "summary": {
                                "total_matches": 0,
                                "omitted_matches": 0,
                                # "matches_by_file": {},
                            },
                            "formatted_output": "No matches found.",
                            # "matches_by_file": [],
                        },
                    )
                return ToolResult(
                    tool_name=self.TOOL_NAME,
                    success=False,
                    errors=f"Ripgrep failed with exit code {e.returncode}: {e.stderr.decode('utf-8')}",
                )

        except Exception as e:
            logger.exception("Error in RipGrepTool")
            return ToolResult(
                tool_name=self.TOOL_NAME,
                success=False,
                errors=str(e),
            )

    @classmethod
    def generate_examples(cls) -> list[tuple["RipGrepTool", ToolResult]]:
        from ..agents.implementations import DemoAgent

        return [
            # Example 1: Basic search with file viewer style output
            (
                cls(
                    calling_agent=DemoAgent(),
                    search_string="function",
                    root_dir="/path/to/project",
                    case_sensitive=True,
                    context_lines=2,
                    max_matches=100,
                ),
                ToolResult(
                    tool_name=cls.TOOL_NAME,
                    success=True,
                    output={
                        "summary": {
                            "total_matches": 2,
                            "omitted_matches": 0,
                        },
                        "formatted_output": (
                            "<FILE>\n"
                            "<PATH>/path/to/project/file1.js</PATH>\n"
                            "<LINES>5</LINES>\n"
                            "<CONTENT>\n"
                            "  5 | function calculateTotal(items) {\n"
                            "</CONTENT>\n"
                            "</FILE>\n"
                            "<FILE>\n"
                            "<PATH>/path/to/project/file2.js</PATH>\n"
                            "<LINES>10</LINES>\n"
                            "<CONTENT>\n"
                            " 10 | function processData(data) {\n"
                            "</CONTENT>\n"
                            "</FILE>"
                        ),
                    },
                ),
            ),
            # Example 2: Search with omitted results
            (
                cls(
                    calling_agent=DemoAgent(),
                    search_string="ERROR",
                    root_dir="/path/to/logs",
                    case_sensitive=False,
                    context_lines=1,
                    max_matches=1,
                ),
                ToolResult(
                    tool_name=cls.TOOL_NAME,
                    success=True,
                    output={
                        "summary": {
                            "total_matches": 1,
                            "omitted_matches": 2,
                        },
                        "formatted_output": (
                            "<FILE>\n"
                            "<PATH>/path/to/logs/app.log</PATH>\n"
                            "<LINES>42</LINES>\n"
                            "<CONTENT>\n"
                            " 42 | [2025-01-29] error: Failed to connect to database\n"
                            "</CONTENT>\n"
                            "</FILE>\n"
                            "\nNote: 2 additional matches were omitted due to the global limit. "
                            "Call the ripgrep tool on specific subdirectories for more results."
                        ),
                    },
                ),
            ),
        ]


if __name__ == "__main__":
    import os
    import asyncio
    import tempfile
    from textwrap import dedent
    from ..agents.implementations import DemoAgent

    async def test():
        """Test suite for RipGrepTool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectories for each test
            dirs = {
                'test1': os.path.join(tmpdir, "test1"),
                'test2': os.path.join(tmpdir, "test2"),
                'test3': os.path.join(tmpdir, "test3"),
                'test4': os.path.join(tmpdir, "test4"),
                'test5': os.path.join(tmpdir, "test5"),
                'test6': os.path.join(tmpdir, "test6")
            }

            for d in dirs.values():
                os.makedirs(d)

            # Test 1: Basic search with non-overlapping matches
            with open(os.path.join(dirs['test1'], "file.txt"), "w") as f:
                f.write(dedent("""
                    line 1
                    line 2
                    match one
                    line 4
                    line 5
                    line 6
                    match two
                    line 8
                    line 9
                """).strip())

            tool = RipGrepTool(
                calling_agent=DemoAgent(),
                search_string="match",
                root_dir=dirs['test1'],
                context_lines=2,
                max_matches=10
            )

            result = await tool.run()
            assert result.success, f"Basic search test failed: {result.errors}"
            assert result.output["summary"]["total_matches"] == 2, "Basic test: Expected 2 matches"
            assert "match one" in result.output["formatted_output"]
            assert "match two" in result.output["formatted_output"]
            print("Test 1 passed!")

            # Test 2: Overlapping context regions
            with open(os.path.join(dirs['test2'], "file.txt"), "w") as f:
                f.write(dedent("""
                    line 1
                    match one
                    line 3
                    match two
                    line 5
                """).strip())

            tool = RipGrepTool(
                calling_agent=DemoAgent(),
                search_string="match",
                root_dir=dirs['test2'],
                context_lines=2,
                max_matches=10
            )

            result = await tool.run()
            assert result.success, f"Overlapping context test failed: {result.errors}"
            match_lines = [s for s in result.output["formatted_output"].split('\n') if 'match' in s]
            assert len(match_lines) == 2, "Should find both match lines"
            assert result.output["summary"]["total_matches"] == 2, "Overlapping test: Expected 2 matches"
            print("Test 2 passed!")

            # Test 3: Max matches limit
            with open(os.path.join(dirs['test3'], "file.txt"), "w") as f:
                f.write("\n".join(f"match {i}" for i in range(10)))

            tool = RipGrepTool(
                calling_agent=DemoAgent(),
                search_string="match",
                root_dir=dirs['test3'],
                context_lines=0,
                max_matches=5
            )

            result = await tool.run()
            assert result.success, f"Max matches test failed: {result.errors}"
            assert result.output["summary"]["total_matches"] == 5, "Max matches test: Expected 5 matches"
            assert result.output["summary"]["omitted_matches"] == 5, "Max matches test: Expected 5 omitted"
            print("Test 3 passed!")

            # Test 4: No matches
            with open(os.path.join(dirs['test4'], "file.txt"), "w") as f:
                f.write("nothing to see here\n")  # Changed content to not include 'match'

            tool = RipGrepTool(
                calling_agent=DemoAgent(),
                search_string="match",
                root_dir=dirs['test4'],
                context_lines=2,
                max_matches=10
            )

            result = await tool.run()
            assert result.success, f"No matches test failed: {result.errors}"
            assert result.output["summary"]["total_matches"] == 0, "No matches test: Expected 0 matches"
            assert "No matches found" in result.output["formatted_output"]
            print("Test 4 passed!")

            # Test 5: Multiple files
            with open(os.path.join(dirs['test5'], "file1.txt"), "w") as f:
                f.write("match in first file\n")
            with open(os.path.join(dirs['test5'], "file2.txt"), "w") as f:
                f.write("match in second file\n")

            tool = RipGrepTool(
                calling_agent=DemoAgent(),
                search_string="match",
                root_dir=dirs['test5'],
                context_lines=1,
                max_matches=10
            )

            result = await tool.run()
            assert result.success, f"Multiple files test failed: {result.errors}"
            assert result.output["summary"]["total_matches"] == 2, "Multiple files test: Expected 2 matches"
            file_sections = result.output["formatted_output"].count("<FILE>")  # Removed the -1
            assert file_sections == 2, f"Multiple files test: Expected 2 file sections, got {file_sections}"
            print("Test 5 passed!")

            # Test 6: Case sensitivity
            with open(os.path.join(dirs['test6'], "file.txt"), "w") as f:
                f.write(dedent("""
                    MATCH upper
                    match lower
                    Match Mixed
                """).strip())

            tool = RipGrepTool(
                calling_agent=DemoAgent(),
                search_string="match",
                root_dir=dirs['test6'],
                context_lines=0,
                max_matches=10,
                case_sensitive=True
            )

            result = await tool.run()
            assert result.success, f"Case sensitivity test failed: {result.errors}"
            assert result.output["summary"]["total_matches"] == 1, "Case sensitivity test: Expected 1 match"
            assert "match lower" in result.output["formatted_output"]
            print("Test 6 passed!")

            print("\nAll tests passed!")

    asyncio.run(test())
