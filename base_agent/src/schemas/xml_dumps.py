# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utility to pretty print XML.
"""

import json
import logging

from typing import Any
from pydantic import BaseModel


logger = logging.getLogger(__name__)


def _is_primitive_type(value: Any) -> bool:
    """Check if a value is a primitive type (for JSON-style array formatting)."""
    return isinstance(value, (int, float, str, bool)) or value is None


def _is_simple_dict(value: dict) -> bool:
    """Check if a dictionary contains only primitive values."""
    return isinstance(value, dict) and all(
        _is_primitive_type(v) for v in value.values()
    )


def _is_pydantic_model(value: Any) -> bool:
    """Check if value is a Pydantic model instance."""
    return isinstance(value, BaseModel)


def _is_list_of_pydantic_models(value: list) -> bool:
    """Check if a list contains only Pydantic model instances."""
    return isinstance(value, list) and all(
        isinstance(item, BaseModel) for item in value if item is not None
    )


def _is_simple_list_of_dicts(value: list) -> bool:
    """Check if a list contains only simple dictionaries."""
    return isinstance(value, list) and all(
        _is_simple_dict(d) for d in value if d is not None
    )


def _format_json_value(value: Any) -> str:
    """Format a value as a JSON string with proper type handling."""
    if isinstance(value, bool):
        return str(value).lower()
    elif value is None:
        return "null"
    elif isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, dict):
        return json.dumps(value, default=_format_json_value)
    return str(value)


def _format_value(value: Any, level: int = 1, indent: int = 2) -> str:
    """Format a value with proper indentation and type handling."""
    if value is None:
        return "null"

    if isinstance(value, bool):
        return str(value).lower()

    if isinstance(value, (int, float)):
        return str(value)

    if isinstance(value, BaseModel):
        # Handle single Pydantic model
        items = []
        for field_name, field_value in value.model_dump().items():
            formatted_value = _format_value(field_value, level + 1, indent).rstrip()
            if "\n" in formatted_value:
                items.append(
                    f"{' ' * (level * indent)}<{field_name}>\n{formatted_value}\n{' ' * (level * indent)}</{field_name}>"
                )
            else:
                items.append(
                    f"{' ' * (level * indent)}<{field_name}>{formatted_value}</{field_name}>"
                )
        return "\n" + "\n".join(items)

    if isinstance(value, (list, tuple)):
        # Empty list - use JSON style
        if not value:
            return "[]"

        # If all items are Pydantic models, format as XML items
        if _is_list_of_pydantic_models(value):
            items = []
            for item in value:
                if item is None:
                    items.append(f"{' ' * (level * indent)}<item>null</item>")
                else:
                    formatted_value = _format_value(item, level + 1, indent).rstrip()
                    if "\n" in formatted_value:
                        items.append(
                            f"{' ' * (level * indent)}<item>\n{formatted_value}\n{' ' * (level * indent)}</item>"
                        )
                    else:
                        items.append(
                            f"{' ' * (level * indent)}<item>{formatted_value}</item>"
                        )
            return "\n" + "\n".join(items)

        # If all items are primitive or simple dicts, use JSON array notation
        # if all(_is_primitive_type(x) for x in value) or _is_simple_list_of_dicts(value):
        #     items = [_format_json_value(x) for x in value]
        #     return f"[{', '.join(items)}]"

        # For mixed content or other complex types, use XML items
        items = []
        for item in value:
            if item is None:
                items.append(f"{' ' * (level * indent)}<item>null</item>")
            else:
                formatted_value = _format_value(item, level + 1, indent).rstrip()
                if "\n" in formatted_value:
                    items.append(
                        f"{' ' * (level * indent)}<item>\n{formatted_value}\n{' ' * (level * indent)}</item>"
                    )
                else:
                    items.append(
                        f"{' ' * (level * indent)}<item>{formatted_value}</item>"
                    )
        return "\n" + "\n".join(items)

    if isinstance(value, dict):
        # Empty dict
        if not value:
            return "{}"

        # Simple dict with only primitive values - use JSON
        # if _is_simple_dict(value):
        #     return json.dumps(value, default=_format_json_value)

        # Complex nested structure - use XML
        items = []
        for k, v in value.items():
            formatted_value = _format_value(v, level + 1, indent).rstrip()
            if "\n" in formatted_value:
                # Multiline value needs special handling
                items.append(
                    f"{' ' * (level * indent)}<{k}>\n{formatted_value}\n{' ' * (level * indent)}</{k}>"
                )
            else:
                # Single line value
                items.append(f"{' ' * (level * indent)}<{k}>{formatted_value}</{k}>")
        return "\n" + "\n".join(items)

    # Default to string representation
    return str(value)


def xml_dumps(instance: dict, root_tag: str | None = None, indent: int = 2) -> str:
    """Convert a Python dictionary to an XML string format suitable for LLM tool arguments.

    Follows best practices:
    - Nested Pydantic models as XML
    - Lists of Pydantic models as XML items
    - JSON arrays for primitive lists: [1, 2, 3]
    - JSON arrays for lists of simple objects: [{"x": 1}, {"y": 2}]
    - XML items for lists of complex objects
    - XML for complex nested structures
    - Simple dictionaries as JSON
    - Proper handling of null values and empty lists

    Args:
        instance: Dictionary to convert
        root_tag: Optional name of the root XML tag (default: None)
        indent: Number of spaces for each indentation level (default: 2)

    Returns:
        XML formatted string following best practices

    Examples:
        >>> # Primitive lists use JSON format
        >>> xml_dumps({"numbers": [1, 2, 3]})
        <numbers>[1, 2, 3]</numbers>

        >>> # Lists of simple objects use JSON format too
        >>> xml_dumps({"points": [{"x": 1}, {"y": 2}]})
        <points>[{"x": 1}, {"y": 2}]</points>

        >>> # Complex object lists use XML format
        >>> xml_dumps({"users": [{"name": "alice", "addresses": ["home"]}]})
        <users>
          <item>
            <name>alice</name>
            <addresses>["home"]</addresses>
          </item>
        </users>

        >>> # Mixed content uses appropriate format for each part
        >>> xml_dumps({
        ...     "name": "test",
        ...     "points": [{"x": 1}, {"y": 2}],
        ...     "config": {"simple": true, "nested": {"complex": true}}
        ... })
        <name>test</name>
        <points>[{"x": 1}, {"y": 2}]</points>
        <config>
          <simple>true</simple>
          <nested>
            <complex>true</complex>
          </nested>
        </config>
    """
    # Build output with root tag
    output = []
    if root_tag:
        output.append(f"<{root_tag}>")

    # Add all fields with proper indentation
    for key, value in instance.items():
        formatted_value = _format_value(
            value, level=2 if root_tag else 1, indent=indent
        ).rstrip()
        if "\n" in formatted_value:
            # Multiline value needs special handling
            if root_tag:
                output.append(f"{' ' * indent}<{key}>{formatted_value}")
                output.append(f"{' ' * indent}</{key}>")
            else:
                output.append(f"<{key}>{formatted_value}")
                output.append(f"</{key}>")
        else:
            # Single line value
            if root_tag:
                output.append(f"{' ' * indent}<{key}>{formatted_value}</{key}>")
            else:
                output.append(f"<{key}>{formatted_value}</{key}>")

    if root_tag:
        output.append(f"</{root_tag}>")

    return "\n".join(output)
