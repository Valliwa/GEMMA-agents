# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""File Visualization Utilities

This module provides utilities for generating file tree visualizations, file views and more.
"""

import hashlib
import logging

from typing import List, Optional, Tuple, Union, Set
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def get_file_view(
    file: Path | str, show_line_numbers: bool = False, show_diagnostics: bool = False
) -> tuple[str | None, str | None]:
    """
    Gets an up-to-date view of a file, reporting its modification time, content
    hash, and LSP diagnostics if an LSP server is running for the file's
    language and rooted at the file's workspace.
    """
    try:
        if isinstance(file, str):
            file = Path(file)

        contents = file.read_text()
        mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        content_hash = hashlib.sha256(contents.encode()).hexdigest()
        lines = contents.splitlines()
        line_count = len(lines)

        file_view = f"<FILE>\n<PATH>{file}</PATH>\n"
        file_view += f"<MTIME>{mtime}</MTIME>\n"
        file_view += f"<HASH>{content_hash}</HASH>\n"
        file_view += f"<LINES>{line_count}</LINES>\n"
        file_view += "<CONTENT>\n"

        if show_line_numbers:
            width = len(str(line_count))
            content_lines = [f"{i+1:>{width}} | {line}" for i, line in enumerate(lines)]
            file_view += "\n".join(content_lines)
        else:
            file_view += contents

        file_view += "\n</CONTENT>\n"

        file_view += "</FILE>\n"
        return file_view, None
    except Exception as e:
        return None, str(e)


async def get_file_edit_view(
    file: Path, edit_diff: str, show_diagnostics: bool = False
) -> str | None:
    """
    Displays the file edit, with LSP diagnostic information if available.
    """
    try:
        if isinstance(file, str):
            file = Path(file)

        contents = file.read_text()
        mtime = datetime.fromtimestamp(file.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        content_hash = hashlib.sha256(contents.encode()).hexdigest()
        lines = contents.splitlines()
        line_count = len(lines)

        file_view = f"<FILE_EDIT>\n<PATH>{file}</PATH>\n"
        file_view += f"<MTIME>{mtime}</MTIME>\n"
        file_view += f"<HASH>{content_hash}</HASH>\n"
        file_view += f"<LINES>{line_count}</LINES>\n"
        file_view += "<EDIT_DIFF>\n"

        file_view += edit_diff

        file_view += "\n</EDIT_DIFF>\n"

        file_view += "</FILE_EDIT>\n"
        return file_view
    except Exception as e:
        logger.error(f"Error getting file edit view: {e}")
        return None


@dataclass
class FileTreeOptions:
    """Configuration options for file tree generation.

    Attributes:
        collapse_threshold: Number of items before a directory is collapsed.
            None means never collapse. Default is 10.
        max_chars: Maximum characters in the output. None means no limit.
            Default is None.
        show_hidden: Whether to include hidden files/directories.
            Default is False.
        sort_by_type: Whether to sort directories before files.
            Default is True.
        exclude_patterns: List of glob patterns to exclude.
            Default is ['.git', '__pycache__', '*.pyc'].
        min_dir_level: Minimum directory level to start collapsing.
            Default is 1 (don't collapse root).
        show_mtime: Whether to show modification time
            Default is False
        show_full_path: Whether to show full file paths instead of just names.
            Default is False
    """

    collapse_threshold: Optional[int] = 20
    max_chars: Optional[int] = None
    show_hidden: bool = False
    sort_by_type: bool = True
    exclude_patterns: List[str] | None = None
    min_dir_level: int = 1
    show_mtime: bool = False
    show_full_path: bool = False  # Added this option

    def __post_init__(self):
        if self.exclude_patterns is None:
            # Default patterns to exclude common nuisance directories and files
            self.exclude_patterns = [
                # Python
                "__pycache__",
                "*.py[cod]",  # Matches .pyc, .pyo, .pyd
                "*.so",
                # Version Control
                ".git",
                ".svn",
                ".hg",
                ".bzr",
                # Node.js
                "node_modules",
                # macOS
                ".DS_Store",
                # Build directories
                "build",
                "dist",
                "*.egg-info",
                # IDE directories
                ".idea",
                ".vscode",
                # Compiled files
                "*.class",  # Java
                "*.o",  # C/C++
                "*.dll",  # Windows
                "*.dylib",  # macOS
                # Temporary files
                "*~",
                "*.swp",
                "*.swo",
                # Coverage reports
                ".coverage",
                "htmlcov",
                # Virtual environments
                "venv",
                ".venv",
                "env",
                ".env",
                # Agent output
                "agent_outputs",
            ]


@dataclass
class FileNode:
    """Represents a file or directory in the tree with token-efficient metadata."""

    name: str
    is_dir: bool
    size: int  # bytes
    perms: str  # unix-style permissions in octal
    mtime: float  # modification timestamp
    children: Optional[List["FileNode"]] = None

    def __post_init__(self):
        if self.children is None and self.is_dir:
            self.children = []

    def add_child(self, child: "FileNode"):
        """Add a child node to this directory node."""
        if self.is_dir:
            self.children.append(child)

    def get_size_summary(self) -> Tuple[int, int, int]:
        """Get total size, file count, and dir count for this node.

        Returns:
            Tuple of (total_size, file_count, dir_count)
        """
        if not self.is_dir:
            return self.size, 1, 0

        total_size = 0
        total_files = 0
        total_dirs = 0

        if self.children:
            for child in self.children:
                size, files, dirs = child.get_size_summary()
                total_size += size
                total_files += files
                total_dirs += dirs

        return total_size, total_files, (total_dirs + 1)

    def format_size(self, size: int) -> str:
        """Format size in bytes to a human-readable format.

        Args:
            size: Size in bytes

        Returns:
            Human readable size string (e.g., "1.5KB", "2.1MB")
        """
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024:
                return f"{size:.1f}{unit}" if unit != "B" else f"{size}B"
            size /= 1024
        return f"{size:.1f}PB"

    def to_string(
        self,
        prefix: str = "",
        max_chars: Optional[int] = None,
        collapse_threshold: Optional[int] = None,
        seen: Optional[Set[int]] = None,
        sort_by_type: bool = True,
        current_level: int = 0,
        min_collapse_level: int = 1,
        show_mtime: bool = False,
        show_full_path: bool = False,
        current_path: Optional[str] = None,
        parent_path: Optional[str] = None,
    ) -> str:
        """Convert node to string representation with optional character limit.

        Args:
            prefix: Indentation prefix for this node
            max_chars: Maximum characters in output
            collapse_threshold: Number of items before collapsing
            seen: Set of seen node IDs (for cycle detection)
            sort_by_type: Whether to sort directories before files
            current_level: Current depth in the tree
            min_collapse_level: Minimum level to start collapsing
            show_mtime: Whether to include the modification time
            show_full_path: Whether to show full file paths instead of just names
            current_path: Current path for this node (used with show_full_path)
            parent_path: Path of the parent node (used with show_full_path)

        Returns:
            String representation of the file tree
        """
        if seen is None:
            seen = set()

        # Detect cycles
        node_id = id(self)
        if node_id in seen:
            return f"{prefix}{self.name} [CYCLE DETECTED]\n"
        seen.add(node_id)

        # Format timestamp in local timezone
        # timestamp = datetime.fromtimestamp(self.mtime).strftime("%Y-%m-%d %H:%M")
        timestamp = datetime.fromtimestamp(self.mtime).strftime("%H:%M")

        # Get size info
        total_size, n_files, n_dirs = self.get_size_summary()

        # Determine display name - full path or just name
        if show_full_path and current_path:
            display_name = current_path
        else:
            display_name = self.name

        # Create current line with directory summary for directories
        if self.is_dir:
            current_line = (
                # f"{prefix}{display_name}/ [0{self.perms}] "
                f"{prefix}{display_name}/ "
                f"({self.format_size(total_size)}, {n_files} files, "
                f"{n_dirs-1} dirs){' ' + timestamp if show_mtime else ''}\n"
            )
        else:
            current_line = (
                # f"{prefix}{display_name} [0{self.perms}] "
                f"{prefix}{display_name} "
                f"{self.format_size(self.size)}{' ' + timestamp if show_mtime else ''}\n"
            )

        if not self.is_dir or not self.children:
            return current_line

        result = [current_line]

        # Determine prefix based on display mode
        if show_full_path:
            # With full paths, we don't indent with spaces
            child_prefix = prefix
        else:
            # Traditional tree view with indentation
            child_prefix = prefix + "  "

        # Sort children if requested
        if sort_by_type:
            sorted_children = sorted(
                self.children, key=lambda x: (not x.is_dir, x.name.lower())
            )
        else:
            sorted_children = sorted(self.children, key=lambda x: x.name.lower())

        # Split into directories and files
        directories = [c for c in sorted_children if c.is_dir]
        files = [c for c in sorted_children if not c.is_dir]

        # Check if we should collapse this directory
        should_collapse = (
            collapse_threshold is not None
            and current_level >= min_collapse_level
            and len(self.children) > collapse_threshold
        )

        if should_collapse:
            summary = (
                f"{child_prefix}[{len(self.children)} items "
                f"({n_files} files, {n_dirs-1} dirs, "
                f"{self.format_size(total_size)} total)] [collapsed]\n"
            )
            result.append(summary)
            return "".join(result)

        # Handle character budget
        if max_chars is not None:
            remaining_chars = max_chars - len(current_line)
            if remaining_chars <= 0:
                return current_line

            # Always show directories first
            for dir_child in directories:
                # Construct child path for full path display
                child_path = f"{current_path}/{dir_child.name}" if show_full_path and current_path else dir_child.name

                dir_repr = dir_child.to_string(
                    child_prefix,
                    remaining_chars,
                    collapse_threshold,
                    seen,
                    sort_by_type,
                    current_level + 1,
                    min_collapse_level,
                    show_mtime,
                    show_full_path,
                    child_path,
                    current_path,
                )
                if len("".join(result)) + len(dir_repr) <= max_chars:
                    result.append(dir_repr)
                    remaining_chars -= len(dir_repr)
                else:
                    break

            # Then show files if space permits
            for file_child in files:
                # Construct child path for full path display
                child_path = f"{current_path}/{file_child.name}" if show_full_path and current_path else file_child.name

                file_repr = file_child.to_string(
                    child_prefix,
                    None,  # No char limit for individual files
                    None,  # No collapse for files
                    seen,
                    sort_by_type,
                    current_level + 1,
                    min_collapse_level,
                    show_mtime,
                    show_full_path,
                    child_path,
                    current_path,
                )
                if len("".join(result)) + len(file_repr) <= max_chars:
                    result.append(file_repr)
                else:
                    break
        else:
            # No limit, show everything
            for dir_child in directories:
                # Construct child path for full path display
                child_path = f"{current_path}/{dir_child.name}" if show_full_path and current_path else dir_child.name

                result.append(
                    dir_child.to_string(
                        child_prefix,
                        None,
                        collapse_threshold,
                        seen,
                        sort_by_type,
                        current_level + 1,
                        min_collapse_level,
                        show_mtime,
                        show_full_path,
                        child_path,
                        current_path,
                    )
                )
            for file_child in files:
                # Construct child path for full path display
                child_path = f"{current_path}/{file_child.name}" if show_full_path and current_path else file_child.name

                result.append(
                    file_child.to_string(
                        child_prefix,
                        None,
                        None,
                        seen,
                        sort_by_type,
                        current_level + 1,
                        min_collapse_level,
                        show_mtime,
                        show_full_path,
                        child_path,
                        current_path,
                    )
                )

        return "".join(result)


def _should_include_path(path: Path, options: FileTreeOptions) -> bool:
    """Check if a path should be included based on options.

    Args:
        path: Path to check
        options: File tree options

    Returns:
        True if path should be included
    """
    name = path.name

    # Check hidden files
    if not options.show_hidden and name.startswith("."):
        return False

    # Check exclude patterns
    if options.exclude_patterns:
        for pattern in options.exclude_patterns:
            if path.match(pattern):
                return False

    return True


def _scan_directory(path: Union[str, Path], options: FileTreeOptions) -> FileNode:
    """Scan a directory and create a FileNode tree.

    Args:
        path: Directory path to scan
        options: File tree options

    Returns:
        Root FileNode of the scanned directory
    """
    path_obj = Path(path)
    stats = path_obj.stat()

    # Convert permissions to octal
    perms = oct(stats.st_mode)[-3:]

    node = FileNode(
        name=path_obj.name or path_obj.anchor,
        is_dir=path_obj.is_dir(),
        size=stats.st_size,
        perms=perms,
        mtime=stats.st_mtime,
    )

    if path_obj.is_dir():
        try:
            for child in path_obj.iterdir():
                if _should_include_path(child, options):
                    try:
                        child_node = _scan_directory(child, options)
                        node.add_child(child_node)
                    except (PermissionError, OSError):
                        continue
        except PermissionError:
            pass

    return node


def create_filetree(
    root_path: Union[str, Path], options: Optional[FileTreeOptions] = None
) -> str:
    """Create a human and LLM-readable file tree visualization.

    This function scans a directory and creates a formatted tree visualization
    that includes file metadata, directory summaries, and optional collapsing
    of large directories. The output is optimized for both human readability
    and machine processing.

    Args:
        root_path: Path to the root directory to visualize
        options: Configuration options for tree generation. If None,
            default options will be used.

    Returns:
        Formatted string representation of the file tree

    Example:
        >>> print(create_filetree('/path/to/project'))
        project/ [0755] (1.2MB, 25 files, 5 dirs) 2025-01-14 15:00
          src/ [0755] (800KB, 15 files, 3 dirs) 2025-01-14 15:00
            main.py [0644] 50KB 2025-01-14 15:00
            utils.py [0644] 30KB 2025-01-14 15:00
          tests/ [0755] (400KB, 10 files, 2 dirs) 2025-01-14 15:00 [collapsed]

        # With show_full_path=True:
        >>> options = FileTreeOptions(show_full_path=True)
        >>> print(create_filetree('/path/to/project', options))
        project/ [0755] (1.2MB, 25 files, 5 dirs) 2025-01-14 15:00
        project/src/ [0755] (800KB, 15 files, 3 dirs) 2025-01-14 15:00
        project/src/main.py [0644] 50KB 2025-01-14 15:00
        project/src/utils.py [0644] 30KB 2025-01-14 15:00
        project/tests/ [0755] (400KB, 10 files, 2 dirs) 2025-01-14 15:00 [collapsed]
    """
    if options is None:
        options = FileTreeOptions()

    try:
        root = _scan_directory(root_path, options)
        path_obj = Path(root_path)

        # Use the root path's name or anchor as the starting path for full path display
        root_name = path_obj.name or path_obj.anchor
        initial_path = root_name if options.show_full_path else None

        return root.to_string(
            max_chars=options.max_chars,
            collapse_threshold=options.collapse_threshold,
            sort_by_type=options.sort_by_type,
            min_collapse_level=options.min_dir_level,
            show_mtime=options.show_mtime,
            show_full_path=options.show_full_path,
            current_path=initial_path,
        )
    except Exception as e:
        return f"Error creating file tree: {str(e)}"

# Update the example in __main__ to demonstrate the new option
if __name__ == "__main__":
    from pathlib import Path

    # Basic usage with defaults (no char limit, auto-collapse directories with >10 items)
    # tree = create_filetree(Path.cwd())
    # print(tree)

    # With full paths
    # options = FileTreeOptions(show_full_path=True)
    # tree = create_filetree(Path.cwd(), options)
    # print(tree)

    # Advanced usage with custom options
    options = FileTreeOptions(
        collapse_threshold=5,  # Collapse directories with more than 5 items
        max_chars=1000,  # Limit output to 1000 characters
        show_hidden=False,  # Skip hidden files and directories
        sort_by_type=True,  # Sort directories first, then files
        show_full_path=True,  # Show full file paths
    )
    tree = create_filetree(Path.cwd(), options)
    print(tree)
