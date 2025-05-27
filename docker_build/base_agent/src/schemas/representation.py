# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Utilities for consistent representation of Pydantic models.
Provides both JSON and XML based formats optimized for LLM readability.
"""

import json
import logging

from enum import Enum
from typing import Type, Dict, Any, Union, get_args, get_origin, Literal, List
from pydantic import BaseModel

from .xml_dumps import xml_dumps
from ..types.common import ArgFormat

logger = logging.getLogger(__name__)


def get_type_info(field: Any) -> str:
    """
    Get human-readable type info for a field.
    Handles both Pydantic fields and schema properties.
    """
    field_type = field.annotation
    parts = []

    # Handle Optional types first
    is_optional = False
    if get_origin(field_type) is Union and type(None) in get_args(field_type):
        is_optional = True
        parts.append("optional")
        field_type = next(t for t in get_args(field_type) if t is not type(None))

    # Handle Literal types
    if get_origin(field_type) is Literal:
        literal_values = get_args(field_type)
        options = ", ".join(f"'{val}'" for val in literal_values)
        parts.append(f"one of [{options}]")
    elif isinstance(field_type, type) and issubclass(field_type, Enum):
        options = ", ".join(f"'{item.name}'" for item in field_type)
        parts.append(f"one of [{options}]")
    else:
        # Get base type
        if get_origin(field_type) is list:
            item_type = get_args(field_type)[0]
            type_str = f"list of {_get_base_type(item_type)}"
        elif get_origin(field_type) is dict:
            key_type, val_type = get_args(field_type)
            type_str = (
                f"dict of {_get_base_type(key_type)} to {_get_base_type(val_type)}"
            )
        else:
            type_str = _get_base_type(field_type)
        parts.append(type_str)

    # Add constraints
    constraints = _get_field_constraints(field)
    if constraints:
        parts.extend(constraints)

    # Handle special cases first
    is_required = not is_optional and field.is_required()

    if (
        hasattr(field.default, "__class__")
        and field.default.__class__.__name__ == "PydanticUndefinedType"
    ) or field.default is Ellipsis:
        # Required field (PydanticUndefined or Ellipsis)
        parts.append("required")
    elif field.default_factory is not None:
        # Show empty container for defaults from factory functions
        if field_type == Dict or get_origin(field_type) is dict:
            parts.append("default: {}")
        elif field_type == List or get_origin(field_type) is list:
            parts.append("default: []")
    elif field.default is not None:
        # Explicit default value
        parts.append(f"default: {_format_default(field.default)}")
    elif is_optional:
        # Optional field without default
        parts.append("default: null")

    # Add description
    if field.description:
        parts.append(field.description)

    return ", ".join(parts)


def _get_base_type(field_type: Type) -> str:
    """Map Python types to schema types."""
    type_map = {
        str: "string",
        int: "integer",
        float: "float",
        bool: "boolean",
        Any: "any",
    }
    # For custom classes, use class name
    if isinstance(field_type, type):
        if issubclass(field_type, BaseModel):
            return field_type.__name__.lower()
        elif issubclass(field_type, Enum):
            return "enum"
    return type_map.get(field_type, str(field_type))


def _get_field_constraints(field: Any) -> list[str]:
    """Extract field constraints as readable strings."""
    constraints = []
    metadata = field.metadata if hasattr(field, "metadata") else []

    constraint_names = {
        "Gt": ("gt", "greater than"),
        "Ge": ("ge", "min"),
        "Lt": ("lt", "less than"),
        "Le": ("le", "max"),
        "MinLength": ("min_length", "min length"),
        "MaxLength": ("max_length", "max length"),
        "MinItems": ("min_items", "min items"),
        "MaxItems": ("max_items", "max items"),
    }

    for item in metadata:
        item_type = type(item).__name__
        if item_type in constraint_names:
            attr_name, label = constraint_names[item_type]
            value = getattr(item, attr_name)
            if value is not None:
                constraints.append(f"{label}: {value}")

    return constraints


def _format_default(value: Any) -> str:
    """Format default values consistently."""
    # Check for PydanticUndefined (required field with no default)
    if (
        hasattr(value, "__class__")
        and value.__class__.__name__ == "PydanticUndefinedType"
    ):
        return "required"

    if isinstance(value, str):
        return f"'{value}'"
    elif isinstance(value, (list, dict)):
        return json.dumps(value)
    elif isinstance(value, Enum):
        return f"'{value.name}'"
    elif callable(value):  # e.g., default_factory
        return "{}"  # Show empty dict/list for factory defaults
    elif value is Ellipsis:
        return "required"  # Explicit handling of Ellipsis
    elif value is None:
        return "null"

    return str(value).lower() if isinstance(value, bool) else str(value)


def get_json_schema_representation(model: Type[BaseModel]) -> str:
    """
    Generate a JSON schema representation focused on LLM readability.
    """
    fields = model.model_fields
    output = []

    for field_name, field in fields.items():
        type_info = get_type_info(field)
        output.append(f'"{field_name}": {type_info}')

    return "{\n    " + ",\n    ".join(output) + "\n}"


def get_xml_schema_representation(
    model: Type[BaseModel], root_tag: str | None = None
) -> str:
    """
    Generate an XML schema representation focused on LLM readability.
    """
    fields = model.model_fields
    # Add angle brackets for root tag if necessary
    output = [f"<{root_tag}>"] if root_tag else []

    for field_name, field in fields.items():
        info = get_type_info(field)
        output.append(f"<{field_name}>{info}</{field_name}>")

    if root_tag:
        output.append(f"</{root_tag}>")
    return "\n".join(output)


def get_schema_representation(
    cls: Type[BaseModel], arg_format: ArgFormat, root_tag: str | None = None
) -> str:
    if arg_format == ArgFormat.JSON:
        return get_json_schema_representation(cls)
    else:
        return get_xml_schema_representation(cls, root_tag=root_tag)


def dumps(
    instance: dict, format: ArgFormat, indent: int, root_tag: str | None = None
) -> str:
    if format == ArgFormat.JSON:
        return json.dumps(instance, indent=indent)
    else:
        return xml_dumps(instance, root_tag=root_tag, indent=indent)
