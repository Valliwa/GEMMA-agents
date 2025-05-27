# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
XML parsing module that follows the 'preserve content, parse structure' principle.
Content between tags is treated as raw text by default unless the guide object
explicitly indicates structural parsing is needed.

This module provides the core XML parsing functionality for tool argument processing,
with a focus on:
1. Content preservation (maintaining exact formatting where needed)
2. Safe parsing (size limits, error recovery)
3. Clear error reporting
4. Performance optimization
"""

import re
import json
import logging

from enum import Enum
from functools import lru_cache
from typing import Optional, List, Any, Type, Union, get_args, get_origin
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Safety limits
MAX_CONTENT_SIZE = 10 * 1024 * 1024  # 10MB max content size
MAX_NESTING_DEPTH = 1000  # Maximum XML nesting depth
MAX_LIST_ITEMS = 1000  # Maximum number of list items to process


class ParsingError(Exception):
    """Base class for XML parsing errors"""

    pass


class ContentTooLargeError(ParsingError):
    """Raised when content exceeds size limits"""

    pass


class NestingTooDeepError(ParsingError):
    """Raised when XML nesting exceeds limits"""

    pass


class ListTooLargeError(ParsingError):
    """Raised when list exceeds item limit"""

    pass


@lru_cache(maxsize=1024)
def clean_content(content: str) -> str:
    """
    Clean content by removing CDATA wrappers.
    Results are cached to improve performance.
    """
    # Safety check
    if len(content) > MAX_CONTENT_SIZE:
        raise ContentTooLargeError(
            f"Content size {len(content)} exceeds limit {MAX_CONTENT_SIZE}"
        )

    # Remove CDATA wrappers
    cdata_pattern = r"<!\[CDATA\[(.*?)\]\]>"
    return re.sub(cdata_pattern, r"\1", content, flags=re.DOTALL)


@lru_cache(maxsize=1024)
def smart_json_parse(value: str) -> Any:
    """
    Try to intelligently parse a value that might be JSON.
    For Any types, try to return the most appropriate Python type.
    Results are cached to improve performance.
    """
    if not isinstance(value, str):
        return value

    # Clean and strip the value
    value = clean_content(value.strip())

    # Early returns for obvious types
    if not value:
        return value
    if value.lower() == "null":
        return None
    if value.lower() in ("true", "false"):
        return value.lower() == "true"

    # Try to parse as JSON
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        # Not valid JSON, return as is
        return value


def convert_value(value: Any, target_type: Type) -> Any:
    """Convert a value to target type, preserving original format where possible"""
    if value is None:
        return None

    # For Optional types, treat empty/whitespace as None
    origin = get_origin(target_type)
    if origin is Union:  # Optional is Union[T, None]
        args = get_args(target_type)
        if type(None) in args:
            if not value or (isinstance(value, str) and not value.strip()):
                return None
            # Extract the actual type from Optional
            real_type = next(arg for arg in args if arg is not type(None))
            return convert_value(value, real_type)

    # For string type, preserve exact content
    if target_type == str:
        return str(value)

    # Handle list types
    if origin == list:
        # Ensure we have a list to work with
        items = value if isinstance(value, list) else [value]

        # Check list size limit
        if len(items) > MAX_LIST_ITEMS:
            raise ListTooLargeError(
                f"List size {len(items)} exceeds limit {MAX_LIST_ITEMS}"
            )

        # Get the item type for the list
        item_type = get_args(target_type)[0] if get_args(target_type) else Any

        # Convert each item based on type
        result = []
        for item in items:
            if item_type == Any:
                # For List[Any], try to infer the best type
                result.append(smart_json_parse(item) if isinstance(item, str) else item)
            else:
                # For typed lists, properly convert each item
                result.append(convert_value(item, item_type))
        return result

    # For all other types, ensure string input is stripped
    if isinstance(value, str):
        value = value.strip()

    try:
        # Handle primitive types
        if target_type == int:
            float_val = float(value)  # Handle "1.0" gracefully
            if float_val.is_integer():
                return int(float_val)
            raise ValueError(f"Value {value} is not an integer")

        if target_type == bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                lower_val = value.lower()
                if lower_val in ("true", "1", "yes", "on", "t", "y"):
                    return True
                if lower_val in ("false", "0", "no", "off", "f", "n"):
                    return False
            raise ValueError(f"Could not convert {value} to bool")

        if target_type == float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value.strip())
            raise ValueError(f"Could not convert {value} to float")

        if isinstance(target_type, type) and issubclass(target_type, Enum):
            try:
                # Try exact match first
                return target_type[value]
            except KeyError:
                # Try case-insensitive match
                upper_value = value.upper()
                for member in target_type:
                    if member.name.upper() == upper_value:
                        return member
                raise ValueError(
                    f"Value must be one of: {[e.name for e in target_type]}"
                )
    except (ValueError, TypeError) as e:
        # Provide detailed error for debugging
        raise ValueError(
            f"Could not convert '{value}' ({type(value)}) to {target_type}: {str(e)}"
        )

    return value

async def extract_nested_content(content: str, field_type: Optional[Type] = None) -> Any:
    """
    Extract content from potentially nested XML structures.
    If the content contains XML tags, attempt to parse as a nested structure.
    Otherwise fall back to smart_json_parse.
    """
    content = clean_content(content)

    # If we have a specific type that's a Pydantic model, parse as nested structure
    if field_type and isinstance(field_type, type) and issubclass(field_type, BaseModel):
        # Recursively parse nested structure
        parsed_dict, warnings = await xml_str_to_dict(content, field_type)
        if parsed_dict:
            return parsed_dict

    # Look for XML-style tags
    if '<' in content and '>' in content:
        # Build a dictionary from XML-style content
        result = {}
        remaining = content
        while remaining:
            # Find next tag
            tag_start = remaining.find('<')
            if tag_start == -1:
                break

            tag_end = remaining.find('>', tag_start)
            if tag_end == -1:
                break

            # Extract tag name
            tag_name = remaining[tag_start + 1:tag_end]
            if not tag_name or tag_name.startswith('/'):
                remaining = remaining[tag_end + 1:]
                continue

            # Find closing tag
            close_tag = f'</{tag_name}>'
            content_end = remaining.find(close_tag, tag_end)
            if content_end == -1:
                break

            # Extract content between tags
            content_start = tag_end + 1
            field_content = remaining[content_start:content_end]

            # Add to result dict
            result[tag_name] = smart_json_parse(field_content)

            # Move past this field
            remaining = remaining[content_end + len(close_tag):]

        if result:
            return result

    # Fall back to smart JSON parsing
    return smart_json_parse(content)

async def extract_list_items(content: str, item_type: Optional[Type] = None) -> List[Any]:
    """Extract items from XML content in either JSON or <item> format"""
    content = clean_content(content)

    # Try parsing as JSON array first
    try:
        items = json.loads(content)
        if isinstance(items, list):
            if len(items) > MAX_LIST_ITEMS:
                raise ListTooLargeError(
                    f"List size {len(items)} exceeds limit {MAX_LIST_ITEMS}"
                )
            return items
    except json.JSONDecodeError:
        pass

    # Look for <item> tags
    items = []
    remaining = content
    while remaining:
        # Find next item tag
        item_start = remaining.find("<item>")
        if item_start == -1:
            break

        # Find end of item
        item_end = remaining.find("</item>", item_start)
        if item_end == -1:
            break

        # Extract item content
        content_start = item_start + len("<item>")
        item_content = remaining[content_start:item_end]

        # Parse item content, potentially handling nested structure
        items.append(await extract_nested_content(item_content, item_type))

        # Move past this item
        remaining = remaining[item_end + len("</item>") :]

        # Check list size limit
        if len(items) > MAX_LIST_ITEMS:
            raise ListTooLargeError(f"List size exceeds limit {MAX_LIST_ITEMS}")

    # If no items found but we have content, treat as single item
    if not items and content.strip():
        return [smart_json_parse(content.strip())]

    return items


def find_field_content(xml_str: str, field_name: str) -> tuple[Optional[str], str]:
    """Find content between field tags and return remaining XML"""
    # Look for opening and closing tags
    open_tag = f"<{field_name}>"
    close_tag = f"</{field_name}>"

    # Try to find the tags
    start = xml_str.find(open_tag)
    if start == -1:
        return None, xml_str

    # Only search for closing tag after the opening tag
    end = xml_str.find(close_tag, start)
    if end == -1:
        return None, xml_str

    # Extract content between tags
    content_start = start + len(open_tag)
    content = xml_str[content_start:end]

    # Size safety check
    if len(content) > MAX_CONTENT_SIZE:
        raise ContentTooLargeError(
            f"Content size {len(content)} exceeds limit {MAX_CONTENT_SIZE}"
        )

    # Remove this field from XML string
    remaining = xml_str[:start] + xml_str[end + len(close_tag) :]

    # Depth safety check - count remaining tags
    tag_count = remaining.count("<") - remaining.count("</")
    if tag_count > MAX_NESTING_DEPTH:
        raise NestingTooDeepError(
            f"XML nesting depth {tag_count} exceeds limit {MAX_NESTING_DEPTH}"
        )

    return content, remaining.strip()


def convert_to_dict(
    raw_data: dict[str, Any], model_cls: Type[BaseModel]
) -> tuple[dict, List[str]]:
    """Convert raw dictionary to typed dictionary using model guidance"""
    fields = {}
    warnings = []

    # Process each field in model
    for field_name, field in model_cls.model_fields.items():
        try:
            if field_name not in raw_data:
                if field.is_required():
                    warnings.append(f"Required field '{field_name}' is missing")
                else:
                    fields[field_name] = field.default
                continue

            # Convert value with detailed error context
            raw_value = raw_data[field_name]
            try:
                fields[field_name] = convert_value(raw_value, field.annotation)
            except (ValueError, TypeError) as e:
                msg = f"Warning for field '{field_name}': {str(e)}"
                warnings.append(msg)
                logger.warning(msg)
                # Use default for optional fields
                if not field.is_required():
                    fields[field_name] = field.default
        except Exception as e:
            # Catch all for field processing
            msg = f"Error processing field '{field_name}': {str(e)}"
            warnings.append(msg)
            logger.error(msg)
            # Continue processing other fields

    return fields, warnings


async def xml_str_to_dict(
    xml_str: str,
    guide_obj: Type[BaseModel],
    root_tag: str = "TOOL_ARGS",
    repair_model: Any = None,  # Kept for API compatibility
) -> tuple[dict | None, str | None]:
    """
    Parse XML into a dict structure using Pydantic model guidance.

    Args:
        xml_str: XML string to parse
        guide_obj: Pydantic model class for type guidance
        root_tag: Expected root tag name (default: TOOL_ARGS)
        repair_model: Kept for API compatibility but not used

    Returns:
        Tuple of (parsed dict or None, warning message or None)
        The warning message will contain details of any parsing issues
        even if a partial result was obtained.
    """
    try:
        # Initial size check
        if len(xml_str) > MAX_CONTENT_SIZE:
            return (
                None,
                f"XML content size {len(xml_str)} exceeds limit {MAX_CONTENT_SIZE}",
            )

        # Ensure root tag
        if f"<{root_tag}>" not in xml_str:
            xml_str = f"<{root_tag}>{xml_str}</{root_tag}>"

        # Extract fields in model-defined order
        fields = {}
        warnings = []
        remaining_xml = xml_str

        for field_name, field in guide_obj.model_fields.items():
            try:
                content, remaining_xml = find_field_content(remaining_xml, field_name)

                if content is not None:
                    # For list types, extract items first
                    origin = get_origin(field.annotation)
                    if origin == list:
                        # Get the type of list items if available
                        args = get_args(field.annotation)
                        item_type = args[0] if args else None
                        items = await extract_list_items(content, item_type)
                        fields[field_name] = convert_value(items, field.annotation)
                    else:
                        # For non-list types, convert directly
                        fields[field_name] = convert_value(content, field.annotation)
                elif field.is_required():
                    warnings.append(f"Required field '{field_name}' not found")
                else:
                    fields[field_name] = field.default

            except (ValueError, TypeError) as e:
                # Type conversion error - continue with default value
                warnings.append(f"Warning for field '{field_name}': {str(e)}")
                if not field.is_required():
                    fields[field_name] = field.default

            except (ContentTooLargeError, NestingTooDeepError, ListTooLargeError) as e:
                # Safety limit error - abort processing
                return None, f"Safety limit exceeded: {str(e)}"

            except Exception as e:
                # Unexpected error - log and continue
                msg = f"Error processing field '{field_name}': {str(e)}"
                warnings.append(msg)
                logger.error(msg)
                if not field.is_required():
                    fields[field_name] = field.default

        # Return partial results if we have any fields
        if fields:
            warning_str = "\n".join(warnings) if warnings else None
            return fields, warning_str
        else:
            # Empty results, still valid.
            return fields, None

    except Exception as e:
        # Catch-all for unexpected errors
        logger.error(f"Unexpected error during XML parsing: {str(e)}")
        return None, f"XML parsing failed: {str(e)}"
