# Self-Improving Coding Agent
# Copyright (c) 2025 Maxime Robeyns
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
The agents module defines the agents that can be composed and called to
construct the broader scaffolding system. An agent might be thought of as a
function in a program in the sense that they can be invoked, invoke other
agents themselves, be composed and so forth - indeed, we maintain a 'callgraph'
of the agent calls in the system.

Individually, an "agent" is just a class that is used to carefully compose an
LLM's context. The way the LLM sees the context is as follows:

- a system prompt section, in which the "agent's" definition, goals, and
  available tools and sub-agents are defined
- the first "user" message, referred to as the core prompt section, which is
  defined by the agent itself and which pertains to the way in which the agent
  should go about its execution; what sequence of steps it should follow, what
  it should focus on, what outcomes it should try to achieve. This is also
  where we put visualisations of system state such as file trees and file
  viewers.
- the "assistant" message, which contains the agent's response and consists of
  alternating sequences of thought and tool or sub-agent calls. The
  'function calling interface' for tools and sub-agents is very similar:
  consisting of an XML sequence whose last closing tag is a stop token. After
  being generated, the LLM will stop, the contents of the XML will be pasrsed
  to identify the tool or sub-agent name, and the arguments provided, these
  will be validated, the tool or sub-agent will be run, and the response will
  be serialised. These will then be concatenated to the previously generated
  assistant message, and the LLM will be called again with this as the
  assistant "pre-fill".

Note, the way this is implemented, and the programming model to maintain, is
that each agent maintains an 'event stream', published to the event bus. This
is a list of events (such as new assistant messages, tool calls and results,
agent calls and results, file events, overseer notifications and so forth)
which describes the exection of the agent. The assistant message is
reconstructed by filtering this event stream and concatenating the values. At a
basic level, just the assistant messages and tool / agent results can be
concatenated, although other event types can be included. For instance, the
file open event may also be included here (with a view of the file content) in
order to save re-generating the core prompt, which would cause a KV cache miss.
By only appending to the LLM agent's context, we can avoid breaking to the
cache, at the cost of lengthening it and potentially duplicating content -
eventually it becomes more cost effective to consolidate all this file state
into the core prompt, shorten the prompt yet re-calculate the KV cache.

Also note that overseer notification events are handled slightly differently.
When reconstructing the event stream, we stop the current assistant message,
add the overseer notification in a new 'user' message, before continuing with
the rest of the events in a new assistant pre-fill message.
"""
