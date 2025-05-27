# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import ast
import json
import shutil
import random
import logging
import tempfile
import warnings
import subprocess

from typing import List, Dict, Optional, ClassVar, Tuple, Set
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import git

from .base import BaseBenchmark, Problem


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set warning format to avoid duplicate messages
warnings.filterwarnings('once')  # Show each warning type only once

# Carefully curated set of repositories for high-quality symbol finding
REPO_LIST: Set[str] = {
    # Web Framework Core - sophisticated dependency injection, middleware
    "pallets/flask",           # Core routing and context management
    "django/django",           # Rich ORM and middleware patterns

    # Data Science Core - complex class hierarchies
    "pandas-dev/pandas",       # DataFrame implementation
    "numpy/numpy",             # Array operations and ufuncs
    "pytorch/pytorch",         # Autograd and tensor operations

    # Code Analysis - AST manipulation, symbol resolution
    "pylint-dev/pylint",      # Static analysis patterns
    "PyCQA/isort",            # Import sorting with complex rules

    # Database Abstractions - complex query building
    "sqlalchemy/sqlalchemy",  # SQL toolkit and ORM
    "encode/databases",       # Async database abstraction

    # # Async Frameworks - concurrency patterns
    "aio-libs/aiohttp",      # Async HTTP client/server
    "samuelcolvin/pydantic", # Data validation with complex typing

    # # Systems Programming - low-level interfaces
    "pyca/cryptography",     # Cryptographic recipes
    "psf/requests",          # HTTP client implementation
    "encode/httpx",          # Modern HTTP client with async
}


@dataclass
class SymbolLocation:
    """Represents the location of a symbol definition"""
    filepath: str
    line: int      # 1-based line number
    column: int    # 0-based column number
    symbol: str    # The actual symbol name
    type: str      # 'function', 'class', 'method', 'variable'
    context: str   # Surrounding code for verification

    def __post_init__(self):
        """Validate the SymbolLocation data"""
        assert isinstance(self.filepath, str) and self.filepath, "Filepath must be non-empty string"
        assert isinstance(self.line, int) and self.line > 0, "Line must be positive integer"
        assert isinstance(self.column, int) and self.column >= 0, "Column must be non-negative integer"
        assert isinstance(self.symbol, str) and self.symbol, "Symbol must be non-empty string"
        assert self.type in {'function', 'class', 'method', 'variable'}, f"Invalid type: {self.type}"
        assert isinstance(self.context, str) and self.context, "Context must be non-empty string"


@dataclass
class SymbolLocationProblem(Problem):
    """Problem subclass for finding symbol definitions"""
    repo_name: str
    symbol: str            # Symbol to locate
    symbol_type: str       # Type of symbol
    file_context: str      # File where symbol is used (not necessarily defined)
    usage_line: int        # Line where symbol is used
    usage_context: str     # Code context around usage
    answer: SymbolLocation # Location of the definition

    def __post_init__(self):
        """Validate the SymbolLocationProblem data"""
        assert isinstance(self.repo_name, str) and self.repo_name, "Repo name must be non-empty string"
        assert isinstance(self.symbol, str) and self.symbol, "Symbol must be non-empty string"
        assert isinstance(self.file_context, str) and self.file_context, "File context must be non-empty string"
        assert isinstance(self.usage_line, int) and self.usage_line > 0, "Usage line must be positive integer"
        assert isinstance(self.usage_context, str) and self.usage_context, "Usage context must be non-empty string"
        assert isinstance(self.answer, SymbolLocation), "Answer must be SymbolLocation instance"

    @classmethod
    def from_symbol_data(cls, symbol_data: Dict, problem_id: str) -> "SymbolLocationProblem":
        """Create a SymbolLocationProblem from symbol data with validation"""
        # Validate required fields
        for field in ['repo_name', 'symbol', 'symbol_type', 'file_context', 'usage_line', 'usage_context', 'answer']:
            assert field in symbol_data, f"Missing required field: {field}"

        # Validate answer structure
        answer_data = symbol_data['answer']
        for field in ['filepath', 'line', 'column', 'context']:
            assert field in answer_data, f"Missing required answer field: {field}"

        # Create rich problem statement with usage context
        statement = (
            f"Find the definition of the {symbol_data['symbol_type']} `{symbol_data['symbol']}` "
            f"in the {symbol_data['repo_name']} repo and "
            f"which is used in file {symbol_data['file_context']} on line {symbol_data['usage_line']}.\n\n"
            f"Usage context:\n```python\n{symbol_data['usage_context']}\n```\n\n"
            f"Return the filepath, line number, and column number where this symbol is defined "
            f"in the format `path/from/repo/root.py:line_no:column_no`.\n"
            f"The line number should be 1-based, and the column number should be 0-based, pointing "
            f"to the first character of the actual symbol name (after 'def', 'class', etc)."
        )

        return cls(
            problem_id=problem_id,
            statement=statement,
            answer=SymbolLocation(
                filepath=answer_data['filepath'],
                line=answer_data['line'],
                column=answer_data['column'],
                symbol=symbol_data['symbol'],
                type=symbol_data['symbol_type'],
                context=answer_data['context']
            ),
            answer_discussion=None,
            repo_name=symbol_data['repo_name'],
            symbol=symbol_data['symbol'],
            symbol_type=symbol_data['symbol_type'],
            file_context=symbol_data['file_context'],
            usage_line=symbol_data['usage_line'],
            usage_context=symbol_data['usage_context']
        )


class SymbolVisitor(ast.NodeVisitor):
    """AST visitor to find symbol definitions and their locations"""

    def __init__(self):
        """Initialize the visitor with clean scope tracking"""
        self.definitions = []
        self._current_class = None  # Track class scope for method detection
        self._current_function = None  # Track function scope for nested functions
        self._scope_depth = 0  # Track nesting level

    def visit_FunctionDef(self, node):
        """Find function and method definitions"""
        # Determine if this is a method (defined within a class) or standalone function
        type_str = 'method' if self._current_class else 'function'

        # Always collect function/method for test coverage
        self.definitions.append({
            'name': node.name,
            'type': type_str,
            'line': node.lineno,
            'column': self._calculate_column(node, type_str),
            'end_line': node.end_lineno,
            'end_col': node.end_col_offset
        })

        # Track scope for nested functions
        prev_function = self._current_function
        self._current_function = node.name
        self._scope_depth += 1
        self.generic_visit(node)
        self._current_function = prev_function
        self._scope_depth -= 1

    def visit_ClassDef(self, node):
        """Find class definitions and provide scoping for methods"""
        # Collect class definition
        self.definitions.append({
            'name': node.name,
            'type': 'class',
            'line': node.lineno,
            'column': self._calculate_column(node, 'class'),
            'end_line': node.end_lineno,
            'end_col': node.end_col_offset
        })

        # Manage class scope for proper method detection
        prev_class = self._current_class
        self._current_class = node.name
        self._scope_depth += 1

        # Visit all class contents to find methods
        self.generic_visit(node)

        # Restore previous scope
        self._current_class = prev_class
        self._scope_depth -= 1

    def visit_Assign(self, node):
        """Find significant variable definitions"""
        for target in node.targets:
            if isinstance(target, ast.Name):
                # Only include module-level variables and constants
                if (not self._current_class and
                    (target.id.isupper() or        # Constants
                     target.id.startswith('__'))):  # Special variables
                    self.definitions.append({
                        'name': target.id,
                        'type': 'variable',
                        'line': node.lineno,
                        'column': self._calculate_column(node, 'variable'),
                        'end_line': node.end_lineno,
                        'end_col': node.end_col_offset
                    })
        self.generic_visit(node)

    def _calculate_column(self, node: ast.AST, type_str: str) -> int:
        """Calculate correct column offset for different types of symbols"""
        if isinstance(node, ast.Assign):
            # For assignments, use the start column of the target
            return node.targets[0].col_offset
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            # For functions and classes, skip the 'def'/'class' keyword
            return node.col_offset + len(type_str.split('_')[0]) + 1
        return node.col_offset


class SymbolLocationBenchmark(BaseBenchmark):
    """Benchmark for evaluating symbol definition location capabilities"""

    name: ClassVar[str] = "symbol_location"

    def __init__(self, seed: int = 1, subset_size: int = 5):
        super().__init__(seed, subset_size)
        self.seed = seed
        self.subset_size = subset_size

        # Use the new benchmark data location convention
        self._dataset_path = (
            Path(__file__).parents[3]  # Go up to project root
            / "benchmark_data"
            / "symbol_location_bench"
            / "symbol_data.json"
        )

        self._problems: Optional[List[SymbolLocationProblem]] = None
        self._problem_map: Dict[str, SymbolLocationProblem] = {}
        self._problems_shuffled = False

        # Initialize benchmark data if missing
        self._ensure_benchmark_data()

    def _filter_symbols_by_quality(self, symbols: List[Dict]) -> List[Dict]:
        """Filter symbols based on quality criteria during curation"""
        filtered_symbols = []

        for symbol in symbols:
            # Skip certain module-level variables
            name = symbol['name']
            if name.startswith('__') and name.endswith('__'):
                continue

            if name in {'__all__', '__version__', '__author__',
                       '__doc__', '__package__', '__path__'}:
                continue

            # Analyze symbol complexity
            is_complex = False
            symbol_type = symbol['type']

            if symbol_type == 'class':
                context = symbol['context']
                # Look for interesting class features
                is_complex = any([
                    'def __init__' in context,  # Has constructor
                    'def __' in context,  # Has special methods
                    '@property' in context,  # Has properties
                    '@abstractmethod' in context,  # Is interface
                    'class Meta:' in context,  # Has metaclass
                ])

            elif symbol_type in ('function', 'method'):
                context = symbol['context']
                # Look for interesting function features
                is_complex = any([
                    '->' in context,  # Has return type annotation
                    '@' in context,  # Has decorators
                    'yield' in context,  # Is generator
                    'async' in context,  # Is async
                    'raise' in context,  # Has error handling
                    'if ' in context,  # Has conditionals
                    'for ' in context,  # Has loops
                    'return' in context and  # Has meaningful return
                    'return None' not in context
                ])

            elif symbol_type == 'variable':
                context = symbol['context'].strip()
                # Skip simple assignments
                if '=' in context:
                    value = context.split('=', 1)[1].strip()
                    # Skip string/number literals and empty collections
                    if value in {'[]', '{}', '()', "''", '""', 'None', 'True', 'False'}:
                        continue
                    if value[0] in '"\'[{(' and value[-1] in '"\']})':
                        continue

                # Complex variable definitions
                is_complex = any([
                    'lambda' in context,  # Lambda functions
                    'for' in context,  # Comprehensions
                    'if' in context,  # Conditional expressions
                    '(' in context and ')' in context,  # Function calls
                    '{' in context and '}' in context and ':' in context,  # Dict literals
                ])

            if is_complex:
                filtered_symbols.append(symbol)

        return filtered_symbols

    def _extract_symbol_data(self, filepath: Path, content: str) -> List[Dict]:
        """Extract symbol definitions and their locations with validation"""
        try:
            # Verify the content is valid Python
            tree = ast.parse(content)
            visitor = SymbolVisitor()
            visitor.visit(tree)

            # Get all lines for context extraction
            lines = content.splitlines()
            assert lines, "Empty file"

            symbols = []
            for definition in visitor.definitions:
                # Validate line numbers
                assert 0 < definition['line'] <= len(lines), f"Invalid line number: {definition['line']}"
                assert definition['end_line'] <= len(lines), f"Invalid end line: {definition['end_line']}"

                # Extract context with bounds checking
                start_line = max(0, definition['line'] - 2)
                end_line = min(len(lines), definition['end_line'] + 1)
                context = '\n'.join(lines[start_line:end_line])

                symbol_data = {
                    'name': definition['name'],
                    'type': definition['type'],
                    'filepath': str(filepath),
                    'line': definition['line'],
                    'column': definition['column'],
                    'context': context
                }

                # Verify the symbol appears in its context
                assert definition['name'] in context, f"Symbol {definition['name']} not found in context"

                symbols.append(symbol_data)

            # Apply quality filtering
            return self._filter_symbols_by_quality(symbols)

        except Exception as e:
            logger.warning(f"Failed to extract symbols from {filepath}: {e}")
            return []

    def _find_symbol_usage(
        self, repo_path: Path, symbol: str, definition: Dict
    ) -> Optional[Dict]:
        """Find an interesting usage of the symbol in another file"""
        try:
            def is_interesting_usage(node, content: str, current_file: str) -> bool:
                """Determine if this usage is interesting enough to include"""
                try:
                    # Ensure we're not in the same file as the definition
                    if current_file == definition['filepath']:
                        return False
                    lines = content.splitlines()

                    # Validate line numbers
                    if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
                        return False

                    # Get context around the usage with bounds checking
                    start_line = max(0, node.lineno - 2)
                    end_line = min(len(lines), node.end_lineno + 1)

                    if start_line >= end_line or end_line > len(lines):
                        return False

                    context = '\n'.join(lines[start_line:end_line])

                    # Look for interesting patterns with additional validation
                    lines_in_context = context.split('\n')
                    has_sufficient_context = len(lines_in_context) >= 3
                    contains_symbol = symbol in context

                    if not (has_sufficient_context and contains_symbol):
                        return False

                    context_start = context.strip()
                    is_import = (
                        context_start.startswith('from') or
                        context_start.startswith('import')
                    )

                    return not is_import

                except (IndexError, AttributeError, ValueError) as e:
                    logger.debug(f"Error in is_interesting_usage: {str(e)}")
                    return False

            # Walk through Python files with error handling
            for file_path in repo_path.rglob("*.py"):
                try:
                    relative_path = str(file_path.relative_to(repo_path))

                    # Skip if this is the definition file
                    if relative_path == definition['filepath']:
                        continue

                    # Read and validate file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    if not content.strip():
                        continue

                    try:
                        tree = ast.parse(content)
                    except SyntaxError as e:
                        logger.warning(f"Syntax error in {file_path}: {str(e)}")
                        continue

                    class UsageVisitor(ast.NodeVisitor):
                        def __init__(self):
                            self.usages = []

                        def visit_Name(self, node):
                            try:
                                if hasattr(node, 'id') and node.id == symbol:
                                    if is_interesting_usage(node, content, relative_path):
                                        self.usages.append(node)
                            except Exception as e:
                                logger.debug(f"Error visiting node: {str(e)}")
                            finally:
                                self.generic_visit(node)

                    # Find usages
                    visitor = UsageVisitor()
                    visitor.visit(tree)

                    if visitor.usages:
                        # Get a random usage and extract context
                        usage = random.choice(visitor.usages)

                        try:
                            lines = content.splitlines()

                            # Validate line numbers
                            if not (0 <= usage.lineno - 2 and usage.end_lineno + 1 <= len(lines)):
                                continue

                            start_line = max(0, usage.lineno - 2)
                            end_line = min(len(lines), usage.end_lineno + 1)

                            context = '\n'.join(lines[start_line:end_line])

                            logger.debug(
                                f"Found usage in {relative_path}:{usage.lineno}, "
                                f"symbol: {symbol}, context: {context}"
                            )

                            result = {
                                'filepath': relative_path,
                                'line': usage.lineno,
                                'context': context
                            }

                            # Validate result
                            if not (symbol in context and result['line'] > 0):
                                continue

                            return result

                        except (IndexError, AttributeError) as e:
                            logger.warning(f"Error extracting context from {file_path}: {str(e)}")
                            continue

                except (IOError, UnicodeDecodeError) as e:
                    logger.warning(f"Error reading {file_path}: {str(e)}")
                    continue

            return None

        except Exception as e:
            logger.warning(f"Failed to find usage for {symbol}: {str(e)}")
            return None

    def _curate_dataset(self, max_symbols_per_repo: int = 100):
        """Create the benchmark dataset by processing repositories"""
        try:
            # Create directory structure
            repos_dir = self._dataset_path.parent / "repos"
            repos_dir.mkdir(parents=True, exist_ok=True)

            all_symbol_data = []

            # Clone and process repositories in parallel
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_repo = {
                    executor.submit(self._clone_repo, repo_name, repos_dir): repo_name
                    for repo_name in REPO_LIST
                }

                for future in as_completed(future_to_repo):
                    repo_name = future_to_repo[future]
                    try:
                        repo = future.result()
                        if not repo:
                            continue

                        repo_symbols = []
                        repo_path = repo.working_dir

                        # Walk through Python files in the repo
                        for file_path in Path(repo_path).rglob("*.py"):
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    content = f.read()

                                # Extract symbol definitions
                                symbols = self._extract_symbol_data(
                                    file_path.relative_to(repo_path),
                                    content
                                )

                                for symbol in symbols:
                                    # Find usage of this symbol
                                    usage = self._find_symbol_usage(
                                        Path(repo_path),
                                        symbol['name'],
                                        symbol
                                    )

                                    if usage:
                                        symbol_data = {
                                            'repo_name': repo_name,
                                            'symbol': symbol['name'],
                                            'symbol_type': symbol['type'],
                                            'file_context': usage['filepath'],
                                            'usage_line': usage['line'],
                                            'usage_context': usage['context'],
                                            'answer': {
                                                'filepath': symbol['filepath'],
                                                'line': symbol['line'],
                                                'column': symbol['column'],
                                                'context': symbol['context']
                                            }
                                        }

                                        # Validate the symbol data
                                        assert symbol_data['symbol'] in symbol_data['answer']['context'], (
                                            f"Symbol {symbol_data['symbol']} not found in definition context"
                                        )
                                        assert symbol_data['symbol'] in symbol_data['usage_context'], (
                                            f"Symbol {symbol_data['symbol']} not found in usage context"
                                        )

                                        repo_symbols.append(symbol_data)

                                        if len(repo_symbols) >= max_symbols_per_repo:
                                            break
                                if len(repo_symbols) >= max_symbols_per_repo:
                                    break

                            except Exception as e:
                                logger.warning(f"Error processing file {file_path}: {str(e)}")
                                continue

                        # Add symbols from this repo
                        all_symbol_data.extend(repo_symbols)
                        logger.info(f"Found {len(repo_symbols)} symbols in {repo_name}")

                    except Exception as e:
                        logger.error(f"Error processing repository {repo_name}: {str(e)}")
                        continue

            # Save the dataset
            self._dataset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._dataset_path, 'w', encoding='utf-8') as f:
                json.dump(all_symbol_data, f, indent=2)

            logger.info(f"Saved {len(all_symbol_data)} symbol examples")

        except Exception as e:
            logger.error(f"Failed to curate dataset: {str(e)}")
            raise RuntimeError(f"Dataset curation failed: {str(e)}")

    def _clone_repo(self, repo_name: str, repos_dir: Path) -> Optional[git.Repo]:
        """Clone repository into the benchmark data directory"""
        try:
            repo_dir = repos_dir / repo_name.split("/")[-1]
            if not repo_dir.exists():
                logger.info(f"Cloning {repo_name}...")
                repo = git.Repo.clone_from(
                    f"https://github.com/{repo_name}.git",
                    repo_dir,
                    depth=None
                )
                assert not repo.bare, "Repository is bare"
                return repo
            else:
                logger.info(f"Repository {repo_name} already exists")
                repo = git.Repo(repo_dir)
                assert not repo.bare, "Repository is bare"
                return repo
        except Exception as e:
            logger.error(f"Failed to clone {repo_name}: {str(e)}")
            return None

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
        """Load problems from the dataset file with validation"""
        try:
            with open(self._dataset_path, 'r') as f:
                dataset = json.load(f)

            self._problems = []
            for idx, symbol_data in enumerate(dataset):
                try:
                    # Validate the symbol data
                    assert all(k in symbol_data for k in [
                        'repo_name', 'symbol', 'symbol_type', 'file_context',
                        'usage_line', 'usage_context', 'answer'
                    ]), "Missing required fields in symbol data"
                    problem_id = f"{symbol_data['repo_name']}_{symbol_data['symbol']}_{idx}"

                    problem_id = re.sub(r"[^a-zA-Z0-9_]", "", f"symbol_{problem_id}")
                    problem_id = problem_id.replace("/", "_")

                    problem = SymbolLocationProblem.from_symbol_data(symbol_data, problem_id)

                    # Additional validation
                    assert problem.symbol in problem.answer.context, (
                        f"Symbol {problem.symbol} not found in answer context"
                    )
                    assert problem.symbol in problem.usage_context, (
                        f"Symbol {problem.symbol} not found in usage context"
                    )

                    self._problems.append(problem)
                    self._problem_map[problem_id] = problem

                except Exception as e:
                    logger.warning(f"Skipping invalid problem: {str(e)}")
                    continue

        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset from {self._dataset_path}: {str(e)}"
            )

    async def setup_problem(
        self, problem: Problem, problem_data_dir: Path, container_name: str
    ) -> None:
        """Get the repository ready for the problem by copying from benchmark data"""
        if not isinstance(problem, SymbolLocationProblem):
            raise ValueError("Problem must be a SymbolLocationProblem instance")

        symbol_problem: SymbolLocationProblem = problem

        # Get repo name from the full repo path
        repo_name = symbol_problem.repo_name.split("/")[-1]

        # Source repository path (from benchmark data directory)
        src_repo_path = self._dataset_path.parent / "repos" / repo_name
        if not src_repo_path.exists():
            raise RuntimeError(f"Source repository not found at {src_repo_path}")

        # Copy repository to working directory
        dest_repo_path = problem_data_dir / repo_name

        # If the directory exists (from a previous run), remove it first
        if dest_repo_path.exists():
            shutil.rmtree(dest_repo_path)

        # Copy the entire repository
        shutil.copytree(src_repo_path, dest_repo_path)

        try:
            # Ensure repo is clean and at expected state
            with subprocess.Popen(
                ["git", "clean", "-fdx"],
                cwd=dest_repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as proc:
                stdout, stderr = proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Failed to clean git repo: {stderr.decode()}"
                    )

            with subprocess.Popen(
                ["git", "reset", "--hard"],
                cwd=dest_repo_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as proc:
                stdout, stderr = proc.communicate()
                if proc.returncode != 0:
                    raise RuntimeError(
                        f"Failed to reset git repo: {stderr.decode()}"
                    )

        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Failed to execute git command: {e.cmd}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to prepare repository: {str(e)}")

    async def score_problem(
        self,
        problem: Problem,
        agent_workdir: str,
        agent_answer_dir: str,
        container_name: str,
    ) -> Tuple[float, Optional[str], Optional[str]]:
        """Score the agent's symbol location by comparing to correct location."""
        if not isinstance(problem, SymbolLocationProblem):
            raise ValueError("Problem must be a SymbolLocationProblem instance")

        try:
            # Read agent's answer
            answer_path = Path(agent_answer_dir) / "answer.txt"
            if not answer_path.exists():
                return 0.0, "No answer file found", None
            answer_text = answer_path.read_text().strip()

            # Parse answer in format: filepath:line:column
            try:
                filepath, line, column = answer_text.split(":")
                line = int(line)
                column = int(column)
            except ValueError:
                return 0.0, "Answer must be in format 'filepath:line:column'", None

            # Compare with correct answer
            correct = problem.answer

            # Normalize paths
            correct_path = str(Path(correct.filepath))
            answer_path = str(Path(filepath))

            # Calculate score components
            path_correct = correct_path == answer_path
            line_correct = correct.line == line
            column_correct = correct.column == column

            # Score based on components
            if path_correct and line_correct and column_correct:
                score = 1.0
                msg = "Perfect match"
            else:
                # Path is worth 0.4, line 0.4, column 0.2
                score = (
                    (0.4 if path_correct else 0.0) +
                    (0.4 if line_correct else 0.0) +
                    (0.2 if column_correct else 0.0)
                )
                # score = (
                #     (0.5 if path_correct else 0.0) +
                #     (0.5 if line_correct else 0.0)
                # )

                components = []
                if not path_correct:
                    components.append(f"Wrong file (expected {correct_path})")
                if not line_correct:
                    components.append(f"Wrong line (expected {correct.line})")
                if not column_correct:
                    components.append(f"Wrong column (expected {correct.column})")

                msg = f"Partial match: {', '.join(components)}"


            return score, None, msg

        except Exception as e:
            return 0.0, f"Error scoring answer: {str(e)}", None


if __name__ == '__main__':
    """Comprehensive tests for the symbol location benchmark"""

    def run_tests():
        """Run all benchmark tests"""

        def test_symbol_location():
            """Test SymbolLocation validation"""
            print("\nTesting SymbolLocation...")

            # Valid case
            loc = SymbolLocation(
                filepath="test.py",
                line=1,
                column=0,
                symbol="test_func",
                type="function",
                context="def test_func():\n    pass"
            )
            assert isinstance(loc, SymbolLocation), "Failed to create valid SymbolLocation"

            # Invalid cases
            try:
                SymbolLocation(
                    filepath="",  # Invalid empty filepath
                    line=1,
                    column=0,
                    symbol="test",
                    type="function",
                    context="test"
                )
                assert False, "Should have rejected empty filepath"
            except AssertionError as e:
                assert "Filepath must be non-empty string" in str(e)

            try:
                SymbolLocation(
                    filepath="test.py",
                    line=0,  # Invalid line number
                    column=0,
                    symbol="test",
                    type="function",
                    context="test"
                )
                assert False, "Should have rejected invalid line number"
            except AssertionError as e:
                assert "Line must be positive integer" in str(e)

            print("SymbolLocation tests passed!")

        def test_problem_creation():
            """Test problem creation and validation"""
            print("\nTesting setup_problem...")

            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a fake repository in the benchmark data directory
                bench_dir = Path(tmpdir) / "benchmark_data" / "symbol_location_bench"
                bench_dir.mkdir(parents=True)
                repo_dir = bench_dir / "repos" / "test-repo"
                repo_dir.mkdir(parents=True)

                # Initialize test repository
                repo = git.Repo.init(repo_dir)

                # Add a test file
                test_file = repo_dir / "test.py"
                test_file.write_text("def test_func():\n    pass\n")

                repo.index.add(['test.py'])
                repo.index.commit("Initial commit")

                # Create problem data
                problem = SymbolLocationProblem(
                    problem_id="test",
                    statement="Test problem",
                    answer=SymbolLocation(
                        filepath="test.py",
                        line=1,
                        column=4,
                        symbol="test_func",
                        type="function",
                        context="def test_func():\n    pass"
                    ),
                    answer_discussion=None,
                    repo_name="owner/test-repo",
                    symbol="test_func",
                    symbol_type="function",
                    file_context="other.py",
                    usage_line=5,
                    usage_context="result = test_func()"
                )

                # Create benchmark instance
                benchmark = SymbolLocationBenchmark()
                benchmark._dataset_path = bench_dir / "symbol_data.json"

                # Test setup_problem
                problem_dir = Path(tmpdir) / "problem_data"
                problem_dir.mkdir()

                # Run setup_problem
                import asyncio
                asyncio.run(benchmark.setup_problem(problem, problem_dir, "test"))

                # Verify repository was copied correctly
                dest_repo = problem_dir / "test-repo"
                assert dest_repo.exists(), "Repository not copied"
                assert (dest_repo / "test.py").exists(), "Test file not copied"
                assert git.Repo(dest_repo), "Not a valid git repository"

            print("Setup problem tests passed!")

        def test_scoring():
            """Test the scoring system"""
            def approx_equal(a: float, b: float, rel_tol: float = 1e-9) -> bool:
                """Compare floats with relative tolerance for testing"""
                return abs(a - b) <= rel_tol * max(abs(a), abs(b))

            # Create a test problem
            problem = SymbolLocationProblem(
                problem_id="test",
                statement="Test problem",
                answer=SymbolLocation(
                    filepath="dir/test.py",
                    line=10,
                    column=4,
                    symbol="test_func",
                    type="function",
                    context="def test_func():\n    pass"
                ),
                answer_discussion=None,
                repo_name="test/repo",
                symbol="test_func",
                symbol_type="function",
                file_context="usage.py",
                usage_line=5,
                usage_context="test_func()"
            )

            benchmark = SymbolLocationBenchmark()

            # Create a temporary directory for testing
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                agent_workdir = tmpdir
                answer_dir = Path(tmpdir) / "answers"
                answer_dir.mkdir()

                # Test perfect match
                with open(answer_dir / "answer.txt", "w") as f:
                    f.write("dir/test.py:10:4")

                score, _, msg = benchmark.score_problem(
                    problem, agent_workdir, str(answer_dir), "test"
                )
                assert approx_equal(score, 1.0), f"Perfect match should score 1.0, got {score}"
                assert msg == "Perfect match", f"Unexpected message: {msg}"

                # Test wrong file
                with open(answer_dir / "answer.txt", "w") as f:
                    f.write("wrong.py:10:4")

                score, _, msg = benchmark.score_problem(
                    problem, agent_workdir, str(answer_dir), "test"
                )
                assert approx_equal(score, 0.6), f"Wrong file should score 0.6, got {score}"
                assert "Wrong file" in msg, f"Message should mention wrong file: {msg}"

                # Test wrong line
                with open(answer_dir / "answer.txt", "w") as f:
                    f.write("dir/test.py:20:4")

                score, _, msg = benchmark.score_problem(
                    problem, agent_workdir, str(answer_dir), "test"
                )
                assert approx_equal(score, 0.6), f"Wrong line should score 0.6, got {score}"
                assert "Wrong line" in msg, f"Message should mention wrong line: {msg}"

                # Test wrong column
                with open(answer_dir / "answer.txt", "w") as f:
                    f.write("dir/test.py:10:0")

                score, _, msg = benchmark.score_problem(
                    problem, agent_workdir, str(answer_dir), "test"
                )
                assert approx_equal(score, 0.8), f"Wrong column should score 0.8, got {score}"
                assert "Wrong column" in msg, f"Message should mention wrong column: {msg}"

                # Test all wrong
                with open(answer_dir / "answer.txt", "w") as f:
                    f.write("wrong.py:20:0")

                score, _, msg = benchmark.score_problem(
                    problem, agent_workdir, str(answer_dir), "test"
                )
                assert approx_equal(score, 0.0), f"All wrong should score 0.0, got {score}"
                assert all(x in msg for x in ["Wrong file", "Wrong line", "Wrong column"]), (
                    f"Message should mention all errors: {msg}"
                )

            print("Scoring system tests passed!")


        def test_symbol_extraction():
            """Test the symbol extraction from code"""
            print("\nTesting symbol extraction...")

            # Create test content
            test_content = """
class TestClass:
    def method_one(self):
        pass

    def method_two(self):
        pass

def test_function():
    x = 1
    return x

CONSTANT_VALUE = 42
"""
            visitor = SymbolVisitor()
            tree = ast.parse(test_content)
            visitor.visit(tree)

            # Verify all symbols were found
            symbols = {d['name']: d['type'] for d in visitor.definitions}
            assert 'TestClass' in symbols, "Failed to find class"
            assert symbols['TestClass'] == 'class', "Wrong type for class"

            assert 'method_one' in symbols, "Failed to find method"
            assert symbols['method_one'] == 'method', "Wrong type for method"

            assert 'test_function' in symbols, "Failed to find function"
            assert symbols['test_function'] == 'function', "Wrong type for function"

            assert 'CONSTANT_VALUE' in symbols, "Failed to find constant"
            assert symbols['CONSTANT_VALUE'] == 'variable', "Wrong type for constant"

            print("Symbol extraction tests passed!")

        def test_setup_problem():
            """Test the setup_problem functionality"""
            print("\nTesting setup_problem...")

            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                # Create a fake repository in the benchmark data directory
                bench_dir = Path(tmpdir) / "benchmark_data" / "symbol_location_bench"
                bench_dir.mkdir(parents=True)
                repo_dir = bench_dir / "repos" / "test-repo"
                repo_dir.mkdir(parents=True)

                # Initialize test repository
                repo = git.Repo.init(repo_dir)

                # Add a test file
                test_file = repo_dir / "test.py"
                test_file.write_text("def test_func():\n    pass\n")

                repo.index.add(['test.py'])
                repo.index.commit("Initial commit")

                # Create problem data
                problem = SymbolLocationProblem(
                    problem_id="test",
                    statement="Test problem",
                    answer=SymbolLocation(
                        filepath="test.py",
                        line=1,
                        column=4,
                        symbol="test_func",
                        type="function",
                        context="def test_func():\n    pass"
                    ),
                    answer_discussion=None,
                    repo_name="owner/test-repo",
                    symbol="test_func",
                    symbol_type="function",
                    file_context="other.py",
                    usage_line=5,
                    usage_context="result = test_func()"
                )

                # Create benchmark instance
                benchmark = SymbolLocationBenchmark()
                benchmark._dataset_path = bench_dir / "symbol_data.json"

                # Test setup_problem
                problem_dir = Path(tmpdir) / "problem_data"
                problem_dir.mkdir()

                import asyncio
                asyncio.run(benchmark.setup_problem(problem, problem_dir, "test"))

                # Verify repository was copied correctly
                dest_repo = problem_dir / "test-repo"
                assert dest_repo.exists(), "Repository not copied"
                assert (dest_repo / "test.py").exists(), "Test file not copied"
                assert git.Repo(dest_repo), "Not a valid git repository"

                print("Setup problem tests passed!")

        # Run all tests
        try:
            test_symbol_location()
            test_problem_creation()
            test_scoring()
            test_symbol_extraction()
            test_setup_problem()
            print("\nAll tests passed successfully!")
        except AssertionError as e:
            print(f"\nTest failed: {str(e)}")
            raise
    run_tests()
