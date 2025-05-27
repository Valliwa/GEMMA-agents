# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Documentation generator optimized for LLM consumption."""

import ast
import yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set


@dataclass
class ModuleStats:
    total_functions: int = 0
    total_classes: int = 0
    documented_functions: int = 0
    documented_classes: int = 0

    @property
    def module_summary(self) -> str:
        parts = []
        undoc_funcs = self.total_functions - self.documented_functions
        undoc_classes = self.total_classes - self.documented_classes

        if self.documented_functions > 0 or undoc_funcs > 0:
            doc_part = f"{self.documented_functions} documented function{'s' if self.documented_functions != 1 else ''}"
            if undoc_funcs > 0:
                doc_part += (
                    f", +{undoc_funcs} other function{'s' if undoc_funcs != 1 else ''}"
                )
            parts.append(doc_part)

        if self.documented_classes > 0 or undoc_classes > 0:
            doc_part = f"{self.documented_classes} documented class{'es' if self.documented_classes != 1 else ''}"
            if undoc_classes > 0:
                doc_part += f", +{undoc_classes} other class{'es' if undoc_classes != 1 else ''}"
            parts.append(doc_part)

        if parts:
            return f" ({'; '.join(parts)})"
        return ""


@dataclass
class MethodInfo:
    name: str
    kind: str  # 'method', 'classmethod', 'staticmethod', 'property'
    docstring: Optional[str]
    signature: Optional[str]
    line_number: int
    is_private: bool = False
    decorators: Set[str] = field(default_factory=set)  # Store all decorators found


@dataclass
class ClassInfo:
    name: str
    docstring: Optional[str]
    methods: Dict[str, MethodInfo]
    nested_classes: Dict[str, "ClassInfo"]
    line_number: int
    is_private: bool = False


@dataclass
class ModuleInfo:
    name: str
    docstring: Optional[str]
    functions: Dict[str, MethodInfo]  # Reusing MethodInfo for consistency
    classes: Dict[str, ClassInfo]
    stats: ModuleStats
    source_file: Path
    line_number: int


def is_complete_docstring(docstring: Optional[str], kind: str) -> bool:
    """
    Determine if a docstring should be included in documentation.

    Args:
        docstring: The docstring to check
        kind: Type of item ('function', 'class', 'module')

    Returns:
        bool: True if docstring should be included in documentation
    """
    if not docstring:
        return False

    # Always include module and class docstrings
    if kind in ("module", "class"):
        return True

    # For functions/methods, require Args: or Returns:
    return "Args:" in docstring or "Returns:" in docstring


class DocumentationGenerator:
    def __init__(self, root_path: Union[str, Path]):
        self.root_path = Path(root_path).resolve()
        self.modules: Dict[str, ModuleInfo] = {}
        self.ignored_dirs = {".git", ".venv", "venv", "__pycache__", "build", "dist"}

    def _get_function_signature(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> str:
        """Extract complete function signature including type hints"""
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.unparse(arg.annotation)}"
            args.append(arg_str)

        returns = f" -> {ast.unparse(node.returns)}" if node.returns else ""
        async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
        return f"{async_prefix}def {node.name}({', '.join(args)}){returns}"

    def _is_top_level_node(self, node: ast.AST, module: ast.Module) -> bool:
        """Check if a node is at the module level"""
        for n in module.body:
            if n is node:
                return True
        return False

    def _process_method(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        decorators: List[ast.expr],
    ) -> Optional[MethodInfo]:
        """
        Process a method definition. Returns None if the method should not be included
        in documentation based on docstring heuristics.
        """
        docstring = ast.get_docstring(node)
        # For methods/functions, require Args: or Returns: in docstring
        if not docstring or not is_complete_docstring(docstring, "function"):
            return None

        kind = "method"
        decorators_found = set()

        for decorator in decorators:
            if isinstance(decorator, ast.Name):
                decorators_found.add(decorator.id)
                if decorator.id == "classmethod":
                    kind = "classmethod"
                elif decorator.id == "staticmethod":
                    kind = "staticmethod"
                elif decorator.id == "property":
                    kind = "property"
                elif decorator.id == "abstractmethod":
                    decorators_found.add("abstract")
            elif isinstance(decorator, ast.Attribute):
                if decorator.attr in ("getter", "setter", "deleter"):
                    kind = "property"
                # Handle abc.abstractmethod and other qualified names
                if decorator.attr == "abstractmethod":
                    decorators_found.add("abstract")
                if decorator.attr == "classmethod":
                    decorators_found.add("classmethod")
                    kind = "classmethod"

        if isinstance(node, ast.AsyncFunctionDef):
            decorators_found.add("async")

        return MethodInfo(
            name=node.name,
            kind=kind,
            docstring=docstring,
            signature=self._get_function_signature(node),
            line_number=node.lineno,
            is_private=node.name.startswith("_"),
            decorators=decorators_found,
        )

    def _process_class(self, node: ast.ClassDef) -> ClassInfo:
        """Process a class definition including nested classes"""
        methods = {}
        nested_classes = {}

        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method = self._process_method(item, item.decorator_list)
                if method is not None:  # Only include methods that pass docstring check
                    methods[item.name] = method
            elif isinstance(item, ast.ClassDef):
                nested_class = self._process_class(item)
                nested_classes[item.name] = nested_class

        return ClassInfo(
            name=node.name,
            docstring=ast.get_docstring(node),
            methods=methods,
            nested_classes=nested_classes,
            line_number=node.lineno,
            is_private=node.name.startswith("_"),
        )

    def _class_to_dict(self, class_info: ClassInfo) -> Dict[str, Any]:
        """Convert ClassInfo to dictionary structure for YAML"""
        result = {}

        # First add methods
        methods = []
        for name, method in sorted(class_info.methods.items()):
            if not method.is_private:
                prefix = ""
                if method.kind == "classmethod":
                    prefix = "@classmethod "
                elif method.kind == "staticmethod":
                    prefix = "@staticmethod "
                elif method.kind == "property":
                    prefix = "@property "
                methods.append(prefix + name)
        if methods:
            result["methods"] = methods

        # Then add nested classes
        nested = {}
        for name, nested_class in sorted(class_info.nested_classes.items()):
            if not nested_class.is_private:
                nested[name] = self._class_to_dict(nested_class)
        if nested:
            result["nested_classes"] = nested

        return result

    def generate_index(self) -> Dict[str, Any]:
        """Generate the module index structure"""
        index = {"modules": {}}

        for name, module in sorted(self.modules.items()):
            module_index = []

            # First add classes
            for class_name, class_info in sorted(module.classes.items()):
                if not class_info.is_private:
                    module_index.append({class_name: self._class_to_dict(class_info)})

            # Then add top-level functions
            for func_name, func_info in sorted(module.functions.items()):
                if not func_info.is_private:
                    module_index.append(func_name)

            if module_index:  # Only add module if it has documented items
                index["modules"][name] = module_index

        return index

    def _process_python_file(self, file_path: Path, rel_path: Path):
        """Process a Python file"""
        try:
            with open(file_path) as f:
                content = f.read()
                module = ast.parse(content)

            stats = ModuleStats()
            functions = {}
            classes = {}

            for node in module.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Count all functions for stats
                    stats.total_functions += 1
                    # Only process functions that pass docstring check
                    method = self._process_method(node, node.decorator_list)
                    if method is not None:
                        stats.documented_functions += 1
                        functions[node.name] = method

                elif isinstance(node, ast.ClassDef):
                    # Count all classes for stats
                    stats.total_classes += 1
                    # For classes, we still process them even without docstrings
                    if ast.get_docstring(node):
                        stats.documented_classes += 1
                    classes[node.name] = self._process_class(node)

            if functions or classes or ast.get_docstring(module):
                # Create module name from full relative path
                module_name = str(rel_path.parent / rel_path.stem)
                if module_name.endswith("/__init__"):
                    module_name = str(rel_path.parent)
                # Replace directory separators with dots
                module_name = module_name.replace("/", ".").replace("\\", ".")

                self.modules[module_name] = ModuleInfo(
                    name=module_name,
                    docstring=ast.get_docstring(module),
                    functions=functions,
                    classes=classes,
                    stats=stats,
                    source_file=file_path,
                    line_number=1,
                )

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    def _get_relative_path(self, path: Path) -> str:
        """Converts an absolute path to a path relative to the project root"""
        try:
            return str(path.relative_to(self.root_path))
        except ValueError:
            return str(path)

    def generate(self) -> str:
        """Generate comprehensive project documentation"""
        # Process Python files
        for py_file in self.root_path.rglob("*.py"):
            if any(ignored in py_file.parts for ignored in self.ignored_dirs):
                continue

            rel_path = py_file.relative_to(self.root_path)
            self._process_python_file(py_file, rel_path)

        return self._compile_documentation()

    def _compile_documentation(self) -> str:
        """Compile the documentation into a structured format"""
        output = []

        # Add header with metadata
        output.extend(
            [
                "# Project Documentation",
                f"Generated: {datetime.now().isoformat()}",
                f"Project: {self.root_path.name}",
                "\nThis documentation includes well-documented public functions and classes.",
                "Functions must have either 'Args:' or 'Returns:' sections to be included.",
                "\n## Quick Reference",
                "```yaml",
                yaml.dump(self.generate_index(), sort_keys=True, allow_unicode=True),
                "```",
            ]
        )

        # Add module documentation
        output.append("\n## Modules")
        for module_name, module in sorted(self.modules.items()):
            if module.docstring or module.functions or module.classes:
                output.append(
                    f"\n### Module: {module_name}{module.stats.module_summary}"
                )
                if module.docstring:
                    output.append(module.docstring.strip())

                # Add class documentation
                for class_name, class_info in sorted(module.classes.items()):
                    if not class_info.is_private:
                        output.append(f"\n#### Class: {class_name}")
                        if class_info.docstring:
                            output.append(class_info.docstring.strip())

                        # Add method documentation
                        for method_name, method in sorted(class_info.methods.items()):
                            if not method.is_private and method.docstring:
                                # Build method type designation
                                type_parts = []
                                if "abstract" in method.decorators:
                                    type_parts.append("Abstract")
                                if method.kind == "classmethod":
                                    type_parts.append("Class")
                                elif method.kind == "staticmethod":
                                    type_parts.append("Static")
                                elif method.kind == "property":
                                    type_parts.append("Property")
                                if "async" in method.decorators:
                                    type_parts.append("Async")

                                type_designation = " ".join(type_parts + ["Method"])

                                output.append(
                                    f"\n##### {type_designation}: {method.signature}"
                                )
                                output.append(
                                    f"{self._get_relative_path(module.source_file)}:{method.line_number}"
                                )
                                output.append(method.docstring.strip())

                # Add function documentation
                for func_name, func in sorted(module.functions.items()):
                    if not func.is_private:
                        func_type = (
                            "Async Function"
                            if "async" in func.decorators
                            else "Function"
                        )
                        output.append(f"\n#### {func_type}: {func.signature}")
                        output.append(
                            f"{self._get_relative_path(module.source_file)}:{func.line_number}"
                        )
                        output.append(func.docstring.strip())

        return "\n".join(output)


def generate_documentation(project_path: Union[str, Path]) -> str:
    """Generate documentation for a project"""
    generator = DocumentationGenerator(project_path)
    return generator.generate()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python documentation.py <project_path>")
        sys.exit(1)

    docs = generate_documentation(sys.argv[1])
    print(docs)
