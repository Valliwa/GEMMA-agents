# Self-Improving Agent System

This directory is the root of a coding agent system, which can solve
complex problems across diverse tasks.

Because the agent system can write code, it can also work on its own code. In
fact, if you are reading this readme now as a coding agent, there is a good
chance that it is because you are working on your own code (or a previous /
later version of your own code). It's important to note in these cases that
this code is not hot-loaded or running at this point: in order to test anything
(e.g. a new tool), you must invoke a new test instance of the agent on some
test task, and inspect its outputs and logs.

## Project Structure

```
agent_code    # usually mounted as your /home/agent/workdir
├── agents
│   ├── agent_calling.py      # functions for making agent calls
│   ├── agent_result.py       # agent return types
│   ├── base_agent.py         # the main BaseAgent class, with core agent loop
│   └── implementations       # sub-agent implementations (includes main, coder, reasoner, etc)
├── benchmarks                # directory of code to load and run benchmarks
├── callgraph                 # agent callgraph for oversight and monitoring
├── events
│   ├── event_bus.py          # defines a global event bus
│   ├── event_bus_utils.py    # utilities for using the event bus (like filtering a certain agent's events)
│   └── event_types.py        # the different event types that can be emitted
├── llm                       # llm calling utilities
├── oversight                 # asynchronous agent oversignt
├── schemas                   # XML and JSON schema utilities
├── tools                     # LLM agent tools
│   ├── edit_tools            # directory of tools relating to file editing
│   └── reasoning_structures  # meta-algorithm structures that guide agent execution
├── utils
│   ├── archive_analysis.py   # for runner to sift through benchmark results
│   ├── documentation.py      # documentation generator
│   ├── filetree.py           # filetree visualisation utils
│   └── stop_tokens.py        # stop token defn's to keep them out of codebase
├── agent_change_log.md       # the meta-agent's "workbook" detailing what was tried and what worked
├── agent.py                  # Agent wrapper class, entrypoint and resource manager
├── description.txt           # description of what's special in this agent
├── __main__.py               # main module-level entrypoint for runner
├── requirements.txt          # agent requirements (to be updated if these change)
└── web_server                # web server for visualisation and monitoring
```

WARNING: it is crucial that you run the project from the parent of the root directory. So, if the root directory (`agent_code` above) is mounted at `/home/agent/workdir`, then you MUST run the project in the `/home/agent` directory, with `python -m workdir.agent ...`

We go through the key modules below one by one:

## Agents Module

### What is a single agent?

An agent is an LLM with tools running in a loop, in an environment.

We define an agent as an instance of the BaseAgent class (see
`agents/base_agent.py`). This has a bunch of class variables (`AGENT_NAME`,
`AGENT_DESCRIPTION`, `AVAILABLE_TOOLS`, `AVAILABLE_AGENTS`, etc) which are
intended to be overridden by concrete sub-agents.

The agent class also has a number of private variables which define the agent's
local state variables (we'll talk more about where the 'state' of the agent
system lies when we cover the event bus).

The LLM used for the agent has a context composed of three main conceptual parts:
1. Agent definition: which includes the system prompt, any agent-loop specific
instructions, and if we're using the 'unconstrained' function-calling
interface, then we insert the tool and agent calling instructions here.
2. Core prompt: this is where the agent's main instructions are put
3. The assistant completions and pre-fill. These consist of a sequence of tool
or agent calls, their response, and assistant CoT. Any notifications or other
interventions can be inserted as user messages.

Note that each agent has considerable flexibility to define how the core prompt
is composed based on dynamic information, and also how the assistant prefill
messages are composed. This prefill section is actually best thought of as a
string view of the filtered event bus, with the events pertaining to that
agent. Through the event bus system, agents can publish events and subscribe to
events from other components. This enables efficient state management, while
optimizing for KV cache usage and context-length. From filtering the event bus,
we can re-construct the open file set, calling agent dialogue, and any other
state in any agent for that matter.

The agent could very well simply be a single long sequence of tool calls, and this may achieve a lot. However,
- over long runs, we need mechanisms to keep the agent on track, avoid loops and repeated work, and clear the context
- it may go through phases of specialised work where carrying around all the context or tool information is unnecessary
- we may want to do some exploratory work, the context and dialogue history from which we don't really care about, just the result

All of the above relate to context management, and this leads to the idea of sub-agents:

### Sub-Agents / Function Calls

The project may use the terms sub-agent and functions interchangeably, since invoking a sub-agent closely resembles a function call.

Due to the global event bus, each agent may be set up to 'inherit' some parts or all of the context of its parent calling agent; such as the initial problem submitted by the user, or the open file set. This allows a sub-agent to pick up from where another one left off (with the open files, dialogue history, tool results etc) and keep making tool calls and so forth before returning a result to the calling agent. For example, imagine doing some exploratory work: the agent could call out to a sub agent that inherits all the context, and return a result, which allows all the exploring to be cut out from extending the context leaving just the result.

The `base_agent_tools.py` module defines some 'functional' methods that facilitate sub-agent calling, including `Complete` (allows a sub-agent to indicate that it has finished and exit the execution loop), ExitAgent (for an anomalous early-exit) and ReturnResult, which allows an agent to return some value to its calling agent.

### Agent Call Graph

Due to the sub-agents resembling function invocations, we can build up a call graph. This is useful for observability, and indeed the callgraph module includes functions that give a high-level textual representation of the current state of the graph. This can be fed into overseers or given back to a calling agent, and gives a good idea of what happened during an agent run. By truncating the contents of the event streams and just summarising their occurrance, we can pick out patterns which let us detect looping, recursion and other pathological patterns.

### Adding a New Agent

The steps for adding a new agent are:

1. Create a new module in the `agents/implementations` directory, and extend the `BaseAgent` class
2. Give the tool a clear and descriptive `AGENT_NAME` using `snake_case`
3. Write a good `AGENT_DESCRIPTION` that will give callers a good idea of what the agent does and when to call it
4. Write a concise and direct `SYSTEM_PROMPT` that sets out the definition, goal and approach of the agent
5. Define the `AVAILABLE_TOOLS` set and `AVAILABLE_AGENTS` set
6. Decide on the agent context inheritance rules in the `INHERITANCE` field; what should it inherit from its parent?
7. Decide whether the agent should have a fileview (file tree and open file viewer) in `HAS_FILEVIEW` and other base variables.
8. Define the `construct_core_prompt` method, using the fields and agent context.
9. Generate one or two good examples for use in the auto-generated documentation
10. Optionally review the way the assistant prefill messages are being composed, or the core execution method is being run, or any other part of the BaseAgent class and overload it.
11. You may choose to directly test the agent (optional):
```python
if __name__ == "__main__":
    import asyncio

    async def main():
        workdir = Path("/tmp/workdir")
        logdir = workdir / "agent_outputs"
        problem = "Find out our public IP address."
        main = MainAgent(workdir=workdir, logdir=logdir, debug_mode=True)
        event_bus = await EventBus.get_instance()
        await event_bus.publish(Event(type=EventType.PROBLEM_STATEMENT, content=problem), main._id)
        result = await main.execute()
        print(result)
        # Now check logdir / answer.txt
        # TODO: add assert based tests here

    asyncio.run(main())
```
Note that it is often not worth adding unit tests for a new agent, since these become brittle. You may write unit tests for the tools that the agent calls, but the agent itself is best tested with end-to-end tests.
12. Make the agent available to at least one other agent to integrate it into the rest of the agent system
13. Do an end-to-end test of the agent, attempting to trigger the subagent with your prompt. Running from the `/home/agent` directory, do
```
python -m workdir.agent -p --debug "<some request that will trigger the subagent>" --workdir /tmp/workdir --logdir /tmp/logdir
```
You MUST make sure that you do not run the agent with a workdir and logdir in the agent code, since the agent often generates files and this creates a mess. Always make sure you run it with a temporary directory as the workdir and logdir.

After running the test agent instance, you should inspect the temporary output directories and test that everything went well. In particular look at the traces in the log directory, and if you ran with the `--debug` flag, the `contexts` directory which will show the system prompt, core prompt and each of the prefills for each of the subagents that ran.
14. You MUST now delete the temporary output directory contents from running the test agent
15. Update any documentation, and you're done.

### Current Approach to Sub Agents

The convention currently being observed in the organisation of the agnts is as follows:

- The MainAgent is just a router. It interprets the problem, decides which agent to initially delegate to, gets the response from this agent and dispatches another sub agent, and so forth.
- The agents dispatched my the MainAgent should be long-running agents. Think of these as individual LLM chat conversations: substantive, generally with no less than a dozen messages, and containing the bulk of the work. From this perspective, the MainAgent is taking the output of an LLM 'chat', and (re-)starting new threads to expand on or verify the results of a previous 'chat', and combining the results.

In summary, you should probably avoid making too many 'low level' tools or sub-agents available to the main agent. Make these available to the sub-agents that the main agent is likely to route the problem to instead, such as the CodingAgent or ProblemSolver.
If there is a new type of problem that is ill-matched to the current long-running agents such as the CodingAgent or ProblemSolver, then you should create a new agent with lots of tools available, and make it available to the main agent.
If you are creating a specialised sub-agent or tool, then it should be made available to only the long-running sub-agents that stand to benefit from it, and certainly not the main agent.

## Tools Module

A tool uses the same 'LLM function calling' interface as we use for agent calls (albeit somewhat simpler).

A tool is just a pydantic class with a name and description as class variables, and zero or more pydantic Fields, each with a type, natural-language description field and optionally constraints for validation. The tool result may contain arbitrary values, yet must be string serialisable. When called, the fields are de-serialised, and the abstract `run() -> ToolResult` method is invoked.

While tools may make LLM calls themselves using the `create_completion` helper function, they tend to involve running code.

### Adding a New Tool

The steps for adding a new tool are:

1. Extend the `BaseTool` class
2. Give the tool a clear and descriptive `TOOL_NAME` using `snake_case`
3. Write a good `TOOL_DESCRIPTION` that will give callers a good idea of what the tool does
4. Give the tool zero, one or more fields of the form `foo: int = Field(..., description="clear field description", ge=0)`. Take field ordering into account since Pydantic maintains this order during serialisation (schema generation), and the order in which they are subsequently generated can be advantageous.
5. Implement the tool's main functionality in the `run` method, making use of the field values (e.g. self.foo)
6. Generate one or two high-quality examples which will serve as few-shot documentation in the system prompt
7. Test the tool. For instance, at the bottom of the simple Calculator tool, we have
```python
if __name__ == "__main__":
    import asyncio
    from ..agents.implementations import DemoAgent

    async def test():
        c = Calculator(calling_agent=DemoAgent(), reasoning="...", expression="2+2")
        result = await c.run()

        assert result.tool_name == Calculator.TOOL_NAME
        assert result.success
        assert result.duration < 0.5
        assert result.output == str(4)
        print("All tests pass!")

    asyncio.run(test())
```
And this can be tested with `python -m workdir.tools.calculator`. Note that you should always run every agent-related code from the top-leve directory to avoid relative import path errors. DO NOT USE TESTING FRMEWORKS like unittest and pytest here. You should just use assert-based tests.
8. Integrate the tool with the rest of the system;
8.1 add it to appropriate toolkits in tools/__init__.py
8.2 make it available to appropriate agents in agents/implementations/*.py in their AVAILABLE_TOOLS sets.
9. Once you're happy with this, you should really do an end-to-end test of the agent, attempting to trigger the tool with your prompt to make sure you've successfully integrated it into the AVAILABLE_TOOLS of at least one agent. From the `/home/agent` directory, do
```
python -m workdir.agent -p "<some request that will trigger the tool>" --workdir /tmp/workdir --logdir /tmp/logdir
```

### Reasoning Structures

While the agent is an LLM running in a loop with tools available, and therefore
doesn't have any hard-coded control flow, it may be convenient to express in
code some high-level 'algorithm' for solving some type of problem. For example,
when researching something, we might want to make sure that we start with a
broad query, and then go depth first on some topics, only stopping when we have
resolved our uncertainty.

A reasoning structure re-uses the tool interface, and essentially contains two
parts:
1. The initialisation, where the reasoning structure 'mode' is activated.
2. The reasoning structure steps, where we step through the sequence of steps

Moreover, there are two base classes for the reasoning structures that you can
extend when implementing a new reasoning structure: `sequential.py` and
`sequential_subagent.py`. Both define a list of steps which specify the
reasoning structure / 'algorithm' itself. The former `sequential.py` will
dynamically insert and subsequently remove a tool in the currently running
agent's available_tools set, to be called at the end of each step. The tool
result will then specify the instructions for the next step. The
`sequential_subagent.py` module will be less transparent, and instead invoke a
separate sub-agent for each individual step. This is perhaps the more reliable
of the two options, however it is more heavy-handed, and requires careful
context sharing between the subsequent agents.

## Benchmarks Module

The benchmarks are an integral part of the systematic evaluation of a new agent.

All benchmarks inherit from BaseBenchmark in order to run nicely as part of the benchmark runner. There are just two abstract methods to implement:
- `problems(self) -> list[Problem]` which loads up a list of problems. Since these benchmarks are relatively small (running each problem takes on the order of minutes) we just return a list here since the memory costs aren't high and we get better simplicity.
- `score_problem(self, problem: Problem, agent_workdir: str, agent_answer_dir: str) -> tuple[float, str | None]` which takes the problem instance, the path to the work directory that the agent had mounted to `/home/agent/workdir` as well as the answer directory in which the agent should have left an `answer.txt` file. To get the submitted answer (if appropriate for the problem) you can do

```python
answer_path = Path(agent_answer_dir) / "answer.txt"
llm_answer = answer_path.read_text().strip()
```

Note that there is a `def setup_problem(self, problem: Problem, problem_data_dir: Path) -> None:` lifecycle hook that is called, and allows the benchmark to perform any required setup in that directory.

## Working on the Agent

The agent code uses relative imports throughout. This means that you will often
need to run things at the root of the project structure. The key thing is to
make sure you're in the *parent* directory of the directory containing the
`agent.py` file.

It is better to write end-to-end tests rather than unit tests with complicated
mocking. We've already described how to run end-to-end tests, running the
following from the `/home/agent` directory
```
python -m workdir.agent -p "<some request that will trigger a code path>" --workdir /tmp/workdir --logdir /tmp/logdir
```
When making improvements to the agent, you should carefully document what you tried, and update on learnings from previous updates in the agent_change_log.md file.

## Project Motivation and Desiderata

**On 'taste' when improving the agent:**

There are many ways in which we could choose to take the project when thinking about new features or directions.

Everything should be viewed through the lens of the answers to the following questions:
a) Is the proposed improvement likely to improve the agent's performance across the benchmark suite?
b) Does the proposed modification stand to reduce the wall-clock running time of the agent?
d) Will the proposed change likely reduce the dollar cost of running the agent?
c) Will the proposed change improve the efficiency with which the agent makes progress towards its solution. Note that 'efficient progress' may not look like a straight line, and may involve periods of exploration.

When answering these questions, don't be myopic, and don't attempt to eke out small efficiency gains, since these can in the long run just end up adding complexity and preclude working on bolder designs. In fact, hold simplicity and elegance as key design desiderata since a) the best solutions are often the most elegant, and b) unbounded complexity will frustrate future attempts to improve the agent.

The hierarchy of simplicity from simplest to most involved when it comes to deciding what to work on is: prompts > tools > reasoning structures > agents > the framework itself.

When in doubt, good things to work on are improving the prompts, tools and the agents.

For tools,
- think about elements of past runs that may have benefitited from a specialised tool or pre-packaged functionality
- were any tools used incorrectly or prompted poorly? In this case an effective update might be to update the tool's description, fields, field descriptions and constraints
- is the tool being made available to appropriate agents? Too few, too many?
- Good tools will patch a deficiency of LLM generation by handing off computational work to something that is better suited. Such deficiencies include left-to-right 'causal' autoregressive generation (which make it hard to write things like diffs), inability to count stuff or do arithmetic, inability to step through some finite state machine.

For agents,
- again, inspect past runs for bad characteristics, and attempt to design agents that avoid these pathologies, while reinforcing the good aspects of past runs
- if an agent is being invoked poorly, can the calling agent's core prompt be tweaked, or the called tool's description be tweaked?
- if an agent is performing poorly, can its core prompt be updated to steer its behaviour in a direction we want? Remember, keep core prompts concise, avoid lists, be direct, and tweak sooner than entirely re-write.
- is the way we're composing the agent's events into pre-fill messages optimal? Does it need to be overloaded in a specific sub-agent?
- perhaps the core loop for an agent needs to be reviewed to enable some domain-specific functionality. However, be careful not to impact other agents with this (i.e. overload don't modify BaseAgent) and consider whether this can't be made part of a tool.

Keep in mind the costs and proportionality of different parts of the system:
- LLM calls dominate this system in terms of cost and latency. Do not spend time working on systems level optimisations (like caching files or tool results) since these are unlikely to yield the benchmark improvements that we want, and are largely wasted effort.
- Again, cache-based optimisations are often a bad idea, focus instead on better tools, better sub-agents, prompt tweaks and other system improvements.

Good things to work on are things that make the coding agent system better at coding:
- improving the mechanics of writing and editing code files (better tools, which are more reliable and fast to use)
- exposing traditional developer tools (ripgrep, tree-sitter, etc) to the agent to make it faster and more efficient. When in doubt, just support Python.
- building reasoning and organisational structures which enable the agent to reliably generate high-quality code
- tools, reasoning structures and sub-agents which increase the speeed with which the agent completes coding tasks
- features which improve the quality of the agent's code: ensuring it is functionally correct, well formatted and structured, as well as being maintainable

Try to identify the simplest and most effective change that you can make: often a targeted prompt update will yield signficant improvement across one component of the system and is relatively cheap and easy to implement.
Adding a new tool is a fairly systematic process that can yield improvements, although try to make tools generalisable across problem domains, and bear in mind that each additional tool documentation section will increase the size of the system prompt (and hence cost of the agent) a bit.

**On high token counts**

The token counts may look surprisingly high. This is because they don't just count the length of the context window of the current agent, but accumulate between each tool call. We're careful to leverage the LLM's KV cache the context up to that point, so that the cost of re-processing these tokens is a fraction of what it initially was.

The upshot is that, unusually high token counts is normal for these agent systems, and you should not seek to cut this down too aggressively as an optimisation. In fact, since conditioning an LLM on some carefully chosen context is as valid a way of changing the LLM's behaviour as fine-tuning the weights, long, carefully composed contexts afford the (sub)agent the opportunity to in-context learn properties of the project being worked on, the way tools and sub-agents are called and so forth, making it a more effective agent for the task at hand.
