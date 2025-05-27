# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Core directed graph implementation for tracking sub-agent calls.

Note, in this module, the terms "agent" and "function" are used interchangeably.
"""

from typing import Dict, Set, List, Optional, Iterator
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class FunctionNode:
    """
    Represents a function execution in the call graph.

    This tracks the essential metadata about a function execution,
    including timing, results, and relationships to other functions.
    """

    # Core identity
    id: str
    name: str
    parent_id: Optional[str] = None
    children: Set[str] = field(default_factory=set)

    # Execution state
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    success: bool | None = None
    error: Optional[str] = None

    # Function-specific data
    args: Dict = field(default_factory=dict)
    result: Optional[str] = None

    # Metrics
    token_count: int = 0
    num_cached_tokens: int = 0
    cost: float = 0.0

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate execution duration if completed."""
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


class CallGraph:
    """
    Directed graph tracking function calls / agent calls.

    The graph maintains parent-child relationships between function
    calls and tracks execution metrics for each function.
    """

    def __init__(self):
        self.nodes: Dict[str, FunctionNode] = {}
        self._root_id: Optional[str] = None

    @property
    def root(self) -> Optional[FunctionNode]:
        """Get the root node if it exists."""
        return self.nodes.get(self._root_id) if self._root_id else None

    def add_node(self, node: FunctionNode) -> None:
        """
        Add a node to the graph.

        If this is the first node, it becomes the root.
        """
        self.nodes[node.id] = node
        if not self._root_id:
            self._root_id = node.id

    def get_node(self, node_id: str) -> Optional[FunctionNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def add_edge(self, from_id: str, to_id: str) -> None:
        """Add a directed edge between nodes."""
        if from_id not in self.nodes or to_id not in self.nodes:
            raise ValueError("Both nodes must exist in the graph")

        self.nodes[from_id].children.add(to_id)
        self.nodes[to_id].parent_id = from_id

    def get_children(self, node_id: str) -> List[FunctionNode]:
        """Get all child nodes of a given node."""
        node = self.nodes.get(node_id)
        if not node:
            return []
        return [self.nodes[child_id] for child_id in node.children]

    def get_ancestors(self, node_id: str) -> List[FunctionNode]:
        """Get all ancestors of a node (parent, parent's parent, etc)."""
        ancestors = []
        current = self.nodes.get(node_id)
        while current and current.parent_id:
            parent = self.nodes.get(current.parent_id)
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        return ancestors

    def get_subtree(self, root_id: str) -> Set[str]:
        """Get all node IDs in the subtree rooted at root_id."""
        subtree = {root_id}
        node = self.nodes.get(root_id)
        if node:
            for child_id in node.children:
                subtree.update(self.get_subtree(child_id))
        return subtree

    def remove_subtree(self, root_id: str) -> None:
        """Remove a node and its entire subtree."""
        subtree = self.get_subtree(root_id)
        for node_id in subtree:
            node = self.nodes.pop(node_id, None)
            if node and node.parent_id:
                parent = self.nodes.get(node.parent_id)
                if parent:
                    parent.children.remove(node_id)

    def iter_bfs(self) -> Iterator[FunctionNode]:
        """Iterate through nodes in breadth-first order."""
        if not self._root_id:
            return

        visited = set()
        queue = [self._root_id]

        while queue:
            node_id = queue.pop(0)
            if node_id not in visited:
                visited.add(node_id)
                node = self.nodes.get(node_id)
                if node:
                    yield node
                    queue.extend(node.children)

    def iter_dfs(self) -> Iterator[FunctionNode]:
        """Iterate through nodes in depth-first order."""
        if not self._root_id:
            return

        visited = set()

        def dfs(node_id: str) -> Iterator[FunctionNode]:
            if node_id not in visited:
                visited.add(node_id)
                node = self.nodes.get(node_id)
                if node:
                    yield node
                    for child_id in node.children:
                        yield from dfs(child_id)

        yield from dfs(self._root_id)

    def find_cycles(self) -> List[List[str]]:
        """Find any cycles in the graph."""
        cycles = []
        visited = set()
        path = []
        path_set = set()

        def dfs(node_id: str) -> None:
            if node_id in path_set:
                cycle_start = path.index(node_id)
                cycles.append(path[cycle_start:] + [node_id])
                return

            if node_id in visited:
                return

            visited.add(node_id)
            path.append(node_id)
            path_set.add(node_id)

            node = self.nodes.get(node_id)
            if node:
                for child_id in node.children:
                    dfs(child_id)

            path.pop()
            path_set.remove(node_id)

        if self._root_id:
            dfs(self._root_id)

        return cycles

    def get_execution_metrics(self) -> Dict:
        """Get overall execution metrics."""
        total_tokens = sum(n.token_count for n in self.nodes.values())
        num_cached_tokens = sum(n.num_cached_tokens for n in self.nodes.values())
        total_cost = sum(n.cost for n in self.nodes.values())

        complete_nodes = [
            n for n in self.nodes.values() if n.started_at and n.completed_at
        ]

        total_duration = (
            sum(
                n.duration_seconds
                for n in complete_nodes
                if n.duration_seconds is not None
            )
            if complete_nodes
            else 0
        )

        successes = sum(1 for n in self.nodes.values() if n.success)
        failures = sum(1 for n in self.nodes.values() if not n.success)

        return {
            "total_functions": len(self.nodes),
            "total_tokens": total_tokens,
            "num_cached_tokens": num_cached_tokens,
            "total_cost": total_cost,
            "total_duration": total_duration,
            "successful_calls": successes,
            "failed_calls": failures,
        }
