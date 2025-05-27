# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Event bus module for centralized event management."""

import json
import asyncio
import logging

from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any, ClassVar
from pydantic import BaseModel, PrivateAttr

from ..types.event_types import EventType, Event, FileOperation, FileEvent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EventEncoder(json.JSONEncoder):
    """JSON encoder for handling special types in event serialization."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (Event, FileEvent)):  # Include FileEvent explicitly
            result = {
                "type": obj.type.value,
                "content": obj.content,
                "metadata": obj.metadata,
                "timestamp": obj.timestamp.isoformat(),
            }
            if isinstance(obj, FileEvent):
                result.update({
                    "operation": obj.operation.value,
                    "path": str(obj.path),
                    "mtime": obj.mtime,
                    "content_hash": obj.content_hash,
                })
            return result
        elif hasattr(obj, "model_dump"):
            # Handle Pydantic models (including Completion, Message, TokenUsage, etc.)
            return obj.model_dump()
        elif hasattr(obj, "__dataclass_fields__"):
            # Handle dataclasses (including ModelInfo)
            return {field: getattr(obj, field) for field in obj.__dataclass_fields__}
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif callable(obj):
            return None
        return super().default(obj)


class EventBus(BaseModel):
    """
    Centralized event bus that serves as the single source of truth for application state.

    Features:
    - Global publish/subscribe system
    - Compositional agent ID based event storage
    - Event provenance tracking
    - Event querying capabilities
    - State persistence
    """

    _instance: ClassVar[Optional["EventBus"]] = None
    _lock: ClassVar[Optional[asyncio.Lock]] = None

    _subscribers: Dict[EventType, List[Callable]] = PrivateAttr(
        default_factory=lambda: defaultdict(list)
    )
    _event_store: Dict[str, List[Event | FileEvent]] = PrivateAttr(default_factory=dict)
    _metadata: Dict[str, str | None] = PrivateAttr(
        default_factory=lambda: {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "last_saved": None,
        }
    )

    class Config:
        arbitrary_types_allowed = True

    def __new__(cls) -> "EventBus":
        raise TypeError(
            "EventBus should not be instantiated directly. "
            "Use 'await EventBus.get_instance()' instead."
        )

    @classmethod
    async def get_instance(cls) -> "EventBus":
        """Get or create the singleton instance.

        Returns:
            The global EventBus instance.
        """
        if not cls._lock:
            cls._lock = asyncio.Lock()

        async with cls._lock:
            if not cls._instance:
                instance = super(EventBus, cls).__new__(cls)
                instance.__init__()
                cls._instance = instance
            return cls._instance

    async def publish(self, event: Event | FileEvent, publisher_id: str) -> None:
        """Publish an event to the bus.

        Args:
            event: The event to publish
            publisher_id: Compositional ID of the publishing agent
        """
        logger.debug(f"New event from {publisher_id}: {event.type}")
        # Add provenance
        event.metadata["publisher_id"] = publisher_id

        # Store event
        if publisher_id not in self._event_store:
            self._event_store[publisher_id] = []
        self._event_store[publisher_id].append(event)

        # Notify subscribers
        event_type = event.type
        for callback in self._subscribers[event_type]:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in event subscriber {callback}: {e}")

    def subscribe(
        self,
        event_type: EventType | set[EventType] | list[EventType] | tuple[EventType],
        callback: Callable,
    ) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: Single EventType or collection of EventTypes to subscribe to
            callback: Async callback function for event handling
        """
        if isinstance(event_type, (set, list, tuple)):
            for et in event_type:
                logger.debug(f"Subscribing {callback} to {et}")
                self._subscribers[et].append(callback)
        else:
            logger.debug(f"Subscribing {callback} to {event_type}")
            self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: EventType | set[EventType] | list[EventType] | tuple[EventType],
        callback: Callable,
    ) -> None:
        """Unsubscribe from events of a specific type.

        Args:
            event_type: Single EventType or collection of EventTypes to unsubscribe from
            callback: The callback to remove
        """
        if isinstance(event_type, (set, list, tuple)):
            for et in event_type:
                if callback in self._subscribers[et]:
                    self._subscribers[et].remove(callback)
        else:
            if callback in self._subscribers[event_type]:
                self._subscribers[event_type].remove(callback)

    def get_events(self, publisher_id: str) -> List[Event]:
        """Get all events published by a specific agent.

        Args:
            publisher_id: Compositional ID of the publishing agent

        Returns:
            List of events from the specified publisher
        """
        return self._event_store.get(publisher_id, [])

    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type across all publishers.

        Args:
            event_type: The type of events to retrieve

        Returns:
            List of events of the specified type
        """
        events = []
        for publisher_events in self._event_store.values():
            events.extend([e for e in publisher_events if e.type == event_type])
        return events

    def get_events_in_chain(self, agent_id: str) -> List[Event]:
        """Get all events in an agent's call chain (parent and children).

        Args:
            agent_id: Compositional ID of the agent

        Returns:
            List of events in the agent's call chain
        """
        events = []
        for publisher_id, publisher_events in self._event_store.items():
            if publisher_id.startswith(agent_id) or agent_id.startswith(publisher_id):
                events.extend(publisher_events)
        return sorted(events, key=lambda e: e.timestamp)

    def clear(self) -> None:
        """Clear all events and subscribers (mainly for testing)."""
        self._event_store.clear()
        self._subscribers.clear()

    async def save_state(self, directory: Path) -> None:
        """Save the current state to disk.

        Args:
            directory: The directory to save state in
        """
        directory.mkdir(parents=True, exist_ok=True)

        # Save metadata
        self._metadata["last_saved"] = datetime.now().isoformat()
        (directory / "metadata.json").write_text(json.dumps(self._metadata, indent=2))

        # Save event store
        event_store_dir = directory / "event_store"
        event_store_dir.mkdir(exist_ok=True)
        for agent_id, events in self._event_store.items():
            agent_dir = event_store_dir / agent_id
            agent_dir.mkdir(exist_ok=True)
            (agent_dir / "events.json").write_text(
                json.dumps(events, indent=2, cls=EventEncoder)
            )

    @staticmethod
    def _deserialize_event(data: dict) -> Event | FileEvent:
        """Deserialize an event from its dictionary representation.

        Args:
            data: Dictionary representation of the event

        Returns:
            Reconstructed Event object
        """
        event_type = EventType(data["type"])
        timestamp = datetime.fromisoformat(data["timestamp"])

        if "operation" in data:  # FileEvent
            return FileEvent(
                type=event_type,
                content=data["content"],
                operation=FileOperation(data["operation"]),
                path=data["path"],
                timestamp=timestamp,
                metadata=data["metadata"],
                mtime=data["mtime"],
                content_hash=data["content_hash"],
            )
        else:
            return Event(
                type=event_type,
                content=data["content"],
                timestamp=timestamp,
                metadata=data["metadata"],
            )

    @classmethod
    async def load_state(cls, directory: Path) -> "EventBus":
        """Load state from disk.

        Args:
            directory: The directory to load state from

        Returns:
            EventBus instance with loaded state
        """
        instance = await cls.get_instance()

        # Load metadata
        metadata_file = directory / "metadata.json"
        if metadata_file.exists():
            instance._metadata = json.loads(metadata_file.read_text())

        # Load event store
        event_store_dir = directory / "event_store"
        if event_store_dir.exists():
            for agent_dir in event_store_dir.iterdir():
                if agent_dir.is_dir():
                    agent_id = agent_dir.name
                    events_file = agent_dir / "events.json"
                    if events_file.exists():
                        events_data = json.loads(events_file.read_text())
                        instance._event_store[agent_id] = [
                            cls._deserialize_event(e) for e in events_data
                        ]

        return instance
