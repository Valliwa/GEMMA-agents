# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .representation import (
    get_schema_representation,
    ArgFormat,
    dumps,
)
from .xml_dumps import xml_dumps
from .xml_parsing import xml_str_to_dict
from .json_parsing import json_str_to_dict


from typing import Type
from pydantic import BaseModel


async def args_str_to_dict(
    tool_args: str, guide_obj: Type[BaseModel], arg_format: ArgFormat, root_tag: str
) -> tuple[dict | None, str | None]:

    # Get schema representation for LLM fixing
    if arg_format == ArgFormat.JSON:
        return await json_str_to_dict(tool_args, guide_obj)
    else:
        tool_args = f"<{root_tag}>\n{tool_args}\n</{root_tag}>"
        return await xml_str_to_dict(tool_args, guide_obj, root_tag=root_tag)
