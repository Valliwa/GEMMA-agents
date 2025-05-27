# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Singleton manager for the global callgraph instance."""

import time
import asyncio
import logging

from typing import Optional, Dict, Any
from datetime import datetime

from .digraph import CallGraph, FunctionNode

logger = logging.getLogger(__name__)


class CallGraphManager:
    """
    Singleton manager for the global callgraph instance.

    This uses the singleton pattern to initialise it once globally and re-use
    the same instance on every subsequent construction of the CallGraphManager
    class.

    This class maintains a single CallGraph instance focused solely on tracking
    agent executions and their relationships.
    """

    _instance: Optional["CallGraphManager"] = None
    _lock: Optional[asyncio.Lock] = None

    def __new__(cls) -> "CallGraphManager":
        raise TypeError(
            "CallGraphManager should not be instantiated directly. "
            "Use 'await CallGraphManager.get_instance()' instead."
        )

    @classmethod
    async def get_instance(cls) -> "CallGraphManager":
        """
        Get or create the singleton instance.

        Returns:
            The global CallGraphManager instance.
        """
        if not cls._lock:
            cls._lock = asyncio.Lock()

        async with cls._lock:
            if not cls._instance:
                cls._instance = super(CallGraphManager, cls).__new__(cls)
                await cls._instance._initialize()
            return cls._instance

    async def _initialize(self) -> None:
        """Initialize the singleton instance."""
        self.graph = CallGraph()
        self.current_function_id: Optional[str] = None
        self._node_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task] = {}
        # Token tracking metrics
        self._token_tracking_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "timeouts": 0,
            "total_wait_time": 0.0,
            "last_failure": None,
            "last_failure_reason": None,
        }

    async def register_agent_task(self, agent_id: str, task: asyncio.Task) -> None:
        """
        Register a running agent task

        Args:
            agent_id: ID of the agent
            task: The asyncio task for the agent
        """
        self._tasks[agent_id] = task

    async def cancel_agent(
        self, agent_id: str, cancel_reason: str | None = None
    ) -> None:
        """
        Cancel a running agent task

        Args:
            agent_id: ID of the agent to cancel
            cancel_reason: Optional reason for cancellation
        """
        if agent_id in self._tasks:
            task = self._tasks[agent_id]
            if not task.done():
                task.cancel()
                # We don't await the task here since it's already being awaited
                # in await_agent_task
                await self.fail_agent(
                    agent_id, cancel_reason or "Cancelled by overseer"
                )

    async def cancel_all_agents(self, cancel_reason: str | None = None) -> None:
        """
        Cancel all running agent tasks in reverse call order (most recent first).

        This ensures that child agents are cancelled before their parents, preventing
        any potential race conditions or orphaned processes. All cancellations are
        performed concurrently within each depth level.

        Args:
            cancel_reason: Optional reason for cancellation that will be recorded
        """
        # async with self._node_lock:

        # Get all nodes and their depths in the call graph
        node_depths: dict[str, int] = {}
        tasks_to_cancel = self._tasks.copy()  # Make a copy since dict will change during cancellation

        for node_id in tasks_to_cancel:
            # Calculate depth by counting edges to root
            depth = 0
            current_node = self.graph.get_node(node_id)
            while current_node and current_node.parent_id:
                depth += 1
                current_node = self.graph.get_node(current_node.parent_id)
            node_depths[node_id] = depth

        # Group nodes by depth
        depth_groups: dict[int, list[str]] = {}
        for node_id, depth in node_depths.items():
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(node_id)

        # Cancel tasks depth by depth (deeper/more recent first)
        for depth in sorted(depth_groups.keys(), reverse=True):
            # Cancel all tasks at this depth concurrently
            cancellation_tasks = [
                self.cancel_agent(
                    node_id,
                    cancel_reason or f"Batch cancellation (depth {depth})"
                )
                for node_id in depth_groups[depth]
            ]

            if cancellation_tasks:
                await asyncio.gather(*cancellation_tasks, return_exceptions=True)

    async def start_agent(
        self, agent_name: str, node_id: str, args: Dict[str, Any]
    ) -> None:
        """
        Start tracking a agent execution.

        Args:
            function_name: Name of the function being executed
            node_id: the unique id of the agent
            args: Function arguments

        Returns:
            str: The generated node ID
        """
        async with self._node_lock:
            node = FunctionNode(
                id=node_id,
                name=agent_name,
                args=args,
            )

            # Add to graph and link to parent if exists
            self.graph.add_node(node)
            if self.current_function_id:
                self.graph.add_edge(self.current_function_id, node_id)

            # Start timing
            node.started_at = datetime.now()

            # Update current function
            self.current_function_id = node_id

    async def complete_agent(
        self,
        node_id: str,
        result: str,
        token_count: int,
        num_cached_tokens: int,
        cost: float,
        success: bool,
    ) -> None:
        """
        Mark a function execution as complete.

        Args:
            node_id: ID of the function node
            result: Function output
            token_count: total number of tokens used
            num_cached_tokens: ...of which were cached tokens
            cost: dollar cost of execution
            success: was the agent run successful
        """
        async with self._node_lock:
            node = self.graph.get_node(node_id)
            if node:
                node.completed_at = datetime.now()
                node.success = success
                node.result = result
                node.token_count = token_count
                node.num_cached_tokens = num_cached_tokens
                node.cost = cost

                # Restore parent as current
                if node.parent_id:
                    self.current_function_id = node.parent_id

                # Remove from task registry if present
                if node_id in self._tasks:
                    del self._tasks[node_id]

    async def fail_agent(self, node_id: str, error: str) -> None:
        """
        Mark a agent execution as failed.

        Args:
            node_id: ID of the function node
            error: Error message
        """
        async with self._node_lock:
            node = self.graph.get_node(node_id)
            if node:
                node.completed_at = datetime.now()
                node.success = False
                node.error = error

                # Restore parent as current
                if node.parent_id:
                    self.current_function_id = node.parent_id

                # Remove from task registry if present
                if node_id in self._tasks:
                    del self._tasks[node_id]

    async def track_tokens(
        self, node_id: str, token_count: int, num_cached_tokens: int, cost: float
    ) -> None:
        """
        Update token count and cost for an agent. This operation is made
        safe with a timeout to prevent hanging.

        Args:
            node_id: ID of the function node
            token_count: Number of tokens used
            num_cached_tokens: Number of cached tokens
            cost: Cost in USD
        """
        self._token_tracking_stats["total_calls"] += 1
        start_time = time.monotonic()

        try:
            # Try to acquire lock with timeout
            async with asyncio.timeout(5):  # 5-second timeout
                async with self._node_lock:
                    wait_time = time.monotonic() - start_time
                    self._token_tracking_stats["total_wait_time"] += wait_time

                    node = self.graph.get_node(node_id)
                    if node:
                        node.token_count += token_count
                        node.num_cached_tokens += num_cached_tokens
                        node.cost += cost
                        self._token_tracking_stats["successful_calls"] += 1

        except asyncio.TimeoutError:
            self._token_tracking_stats["timeouts"] += 1
            self._token_tracking_stats["last_failure"] = datetime.now()
            self._token_tracking_stats["last_failure_reason"] = "Timeout acquiring lock"
            logger.warning(f"Timeout while trying to track tokens for node {node_id}")
            # We explicitly don't raise here to prevent hanging the agent
            return

        except Exception as e:
            self._token_tracking_stats["failed_calls"] += 1
            self._token_tracking_stats["last_failure"] = datetime.now()
            self._token_tracking_stats["last_failure_reason"] = str(e)
            logger.error(f"Error tracking tokens for node {node_id}: {e}")
            # We explicitly don't raise here to prevent hanging the agent
            return

    async def get_token_tracking_health(self) -> Dict[str, Any]:
        """Get health metrics about token tracking operations."""
        return {
            "stats": self._token_tracking_stats.copy(),
            "success_rate": (
                self._token_tracking_stats["successful_calls"]
                / self._token_tracking_stats["total_calls"]
                if self._token_tracking_stats["total_calls"]
                else 1.0
            ),
            "average_wait_time": (
                self._token_tracking_stats["total_wait_time"]
                / self._token_tracking_stats["total_calls"]
                if self._token_tracking_stats["total_calls"]
                else 0.0
            ),
        }

    async def get_execution_metrics(self) -> Dict[str, Any]:
        """Get overall execution metrics from the graph."""
        async with self._node_lock:
            return self.graph.get_execution_metrics()

    def clear(self) -> None:
        """Clear the graph state (mainly for testing)."""
        self.graph = CallGraph()
        self.current_function_id = None
        self._token_tracking_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "timeouts": 0,
            "total_wait_time": 0.0,
            "last_failure": None,
            "last_failure_reason": None,
        }
