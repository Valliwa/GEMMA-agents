# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A special-case reasoner agent

Note: this could also be implemented as a reasoning structure or a tool...
"""

import logging

from pathlib import Path
from pydantic import Field
from datetime import datetime

from ..base_agent import BaseAgent
from ...events import EventBus
from ...config import settings
from ...events.event_bus_utils import get_problem_statement
from ...llm.api import create_completion
from ...llm.base import Message
from ...callgraph.manager import CallGraphManager
from ...utils.metrics import make_random_agent_metrics
from ...types.agent_types import AgentStatus, AgentResult, InheritanceFlags, InheritanceConfig
from ...types.event_types import EventType, Event
from ...types.llm_types import ReasoningEffort, TextContent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReasoningAgent(BaseAgent):
    """
    A special-purpose reasoning agent based off reasoning models.
    """

    AGENT_NAME = "reasoning_agent"

    AGENT_DESCRIPTION = """
Call this agent when you need some problem solving to be done: for instance drawing up a plan, coming up with a solution, or just looking over what you have and picking a path forward.

Whenever there is some uncertainty about what to do or hard thinking to be done, call this agent.

This agent will inherit all the open files and your previous message history from your context, so you can make direct reference to that when directing it.

Note however that this agent cannot see the full details of other sub-agents' dialogue histories, the files they opened and so forth. To that extent, make sure you have provided comprehsntive, plentiful and accurate context for it.

Also note that this agent cannot open any files, make file edits, or call any tools for that matter. It merely reasons over the context.

Example capabilities
- Thinking hard
- Solving problems
- Qualitative reasoning

Choose when:
- You've opened some files or gathered some material and now need to reflect on it with a view to solving some problem.
- You'd like some outside opinion on a question, problem or topic

Avoid when:
- The decision trivial in nature, and doing full reasoning would be unwarrented. There is a (small, yet not negligible) cost to calling this agent.
"""

    SYSTEM_PROMPT = """You are a competent and dilligent reasoner and problem solver, working as part of an agent system. Your job is to look over the context provided to you, and find solutions swiftly and effectively. Prefer elegant solutions to complex ones. You cannot call any tools or have any direct effect on the state of things. You are merely here to offer council, advice, solutions and direction to your calling agent.

You do not have access to any tools. Even if you can see tool instructions from the past conversation history, you MUST NOT attempt to call them.
"""

    # Available tools - none
    AVAILABLE_TOOLS = set()

    # Available agents - none
    AVAILABLE_AGENTS = set()

    MODEL = settings.REASONING_MODEL

    HAS_FILEVIEW = True

    INHERITANCE = InheritanceConfig(
        flags=InheritanceFlags.OPEN_FILES | InheritanceFlags.CORE_DIALOGUE
    )

    # Agent parameters
    problem_to_solve: str = Field(..., description="The problem or request to handle")
    share_main_problem_statement: bool = Field(
        default=True,
        description="Whether to share the original problem statement verbatim with the reasoner? Often you will want to set this to 'false' to keep the reasoning focused, but if you are reasoning about the original problem itself, you may want to set this to 'true'.",
    )

    def __init__(self, parent: BaseAgent | None = None, workdir: Path | None = None, logdir: Path | None = None, debug_mode: bool = False, **data):
        super().__init__(parent=parent, workdir=workdir, logdir=logdir, debug_mode=debug_mode, **data)

    async def construct_system_prompt(self) -> str:
        """
        Constructs the system prompt

        Returns:
            The constructed system prompt, ready for use in the agent.
        """
        event_bus = await EventBus.get_instance()

        sys_prompt = self.SYSTEM_PROMPT

        event_bus = await EventBus.get_instance()
        await event_bus.publish(
            Event(type=EventType.SYSTEM_PROMPT_UPDATE, content=sys_prompt),
            self._id,
        )

        # If debugging, save the system prompt to disk
        if self._debug_mode:
            all_events = event_bus.get_events(self._id)
            system_prompts = [
                e for e in all_events if e.type == EventType.SYSTEM_PROMPT_UPDATE
            ]
            sys_path = (
                self._logdir
                / "contexts"
                / f"{self._id}_system_prompt_{len(system_prompts):01d}.txt"
            )
            sys_path.parent.mkdir(exist_ok=True, parents=True)
            sys_path.write_text(sys_prompt)

        return sys_prompt

    async def construct_core_prompt(self) -> str:
        """Construct the core prompt for reasoner solving. Need to implement
        this separately to the overloaded construct_core_prompt because this is
        an abstract method.
        """
        parts = []

        parts.append("""You are a competent and dilligent reasoner and problem solver, working as part of an agent system. Your job is to look over the context provided to you, and find solutions swiftly and effectively. Prefer elegant solutions to complex ones. You cannot call any tools or have any direct effect on the state of things. You are merely here to offer council, advice, solutions and direction to your calling agent.""")

        if self.share_main_problem_statement:
            initial_problem = await get_problem_statement()
            if initial_problem and initial_problem != "":
                parts.append(
                    f"""Here is the initial request posed by the user, provided to the entrypoint of the agent system, and provided to you, the reasoning agent, for context only:

Initial request ================================================================

{initial_problem}

End initial request ============================================================"""
                )

        parts.append(
            f"""Here is the problem that your calling agent has asked to you to reason about specifically:

Problem to think about ========================================================

{self.problem_to_solve}

End problem statement =========================================================

"""
        )

        message_stream = ""
        core_event_types = {
            EventType.ASSISTANT_MESSAGE,
            EventType.TOOL_RESULT,
            EventType.AGENT_RESULT,
        }
        messages = await self.get_assistant_prefill_messages(
            prefill_message_types=core_event_types, for_agent_id=self._parent_id,
        )
        message_stream = "\n".join([str(m) for m in messages])
        if message_stream != "":
            parts.append(
                f"The dialogue history of your parent agent up to this point is:\n {message_stream}"
            )

        parts.append("""Remember, you do not have any tools available, and you are not able to take actions. You task is to reason about the problem that you have specifically been asked to think about, taking into account the initial problem statement if provided and relevant to loosely guide your reasoning. And you should return a solution, guidance, direction or other appropriate resolution to your specific problem statement.""")

        return "\n\n".join(parts)

    async def execute(self) -> AgentResult:
        """
        Executes this agent with parameter values provided in the fields

        Returns:
            The result of the reasoning run, as an AgentResult
        """
        callgraph = await CallGraphManager.get_instance()
        event_bus = await EventBus.get_instance()

        try:
            # Construct the core and system prompts (this saves them to cache)
            # (This also saves it in the context if cildren inherit it)
            system_prompt = await self.construct_system_prompt()
            core_prompt = await self.compose_core_prompt()

            # Single iteration
            messages = [
                Message(role="system", content=[TextContent(text=system_prompt)]),
                Message(role="user", content=[TextContent(text=core_prompt)]),
            ]

            completion = await create_completion(
                messages=messages,
                model=self.MODEL,  # uses reasoner models
                reasoning_effort=ReasoningEffort.HIGH,  # TODO: make configurable
                # No temp, stops, continuations, tools etc for reasoning agent
            )

            self._update_metrics(completion)

            # Track incremental cost - useful for oversight
            await callgraph.track_tokens(
                node_id=self._id,
                token_count=completion.usage.total_tokens,
                num_cached_tokens=completion.usage.cached_prompt_tokens,
                cost=completion.calculate_cost(),
            )

            result = ""
            for block in completion.content:
                if isinstance(block, TextContent):
                    result += block.text

            # if self.return_reasoning:
            #     reasoning = completion.raw_response.get("reasoning_content", "")
            #     result = (reasoning + "\n\n" + content).lstrip()

            await event_bus.publish(
                Event(
                    type=EventType.ASSISTANT_MESSAGE,
                    content=result,
                    metadata=dict(completion=completion),
                ),
                self._id,
            )

            self._metrics.end_time = datetime.now()

            return AgentResult(
                agent_name=self.AGENT_NAME,
                status=AgentStatus.SUCCESS,
                result=result,
                metrics=self._metrics,
            )

        except Exception as e:
            # Handle errors
            self._metrics.end_time = datetime.now()
            logger.info(f"Agent failed: {e}")
            raise e

    @classmethod
    def generate_examples(cls) -> list[tuple["BaseAgent", AgentResult]]:
        """Generate example uses of the tool with their expected outputs."""
        examples = [
            (
                cls(
                    problem_to_solve="""Look at the performance data and the implementation in the open files, and come up with a high-quality code improvement that we should focus on making.""",
                ),
                AgentResult(
                    agent_name=cls.AGENT_NAME,
                    status=AgentStatus.SUCCESS,
                    result="""... just the answer without reasoning output...""",
                    metrics=make_random_agent_metrics(
                        tools_enabled=False, agents_enabled=False
                    ),
                ),
            ),
            (
                cls(
                    problem_to_solve="""Look at the original request, and tell me whether the empirical command execution results get us any closer to solving the problem.""",
                    share_main_problem_statement=True,
                    # return_reasoning=True,
                ),
                AgentResult(
                    agent_name=cls.AGENT_NAME,
                    status=AgentStatus.SUCCESS,
                    result="""... reasoning output followed by reasoner's answer ...""",
                    metrics=make_random_agent_metrics(
                        tools_enabled=False, agents_enabled=False
                    ),
                ),
            ),
        ]
        return examples
