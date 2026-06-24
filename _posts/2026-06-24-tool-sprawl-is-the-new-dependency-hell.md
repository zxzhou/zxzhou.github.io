---
layout: post
title: "Tool Sprawl Is the New Dependency Hell"
date: 2026-06-24
tags: [ai-agents, agent-harness, mcp, context-engineering]
excerpt: "Tool access can make an agent stronger or more distracted. Every tool description, rule file, memory item, and raw result competes for the same context budget."
---

Agent builders like adding tools.

The impulse is understandable. A model with no tools can only talk. A model with a shell can inspect a system. A model with a browser can verify behavior. A model with database access can answer questions grounded in live state. A model with MCP servers can reach across many applications.

The next assumption is where things go wrong: more tools means more agency.

That is true only up to a point. After that, tool access becomes tool sprawl. The agent has more verbs, but less attention. It has more possible paths, but weaker judgment about which path matters. It has more context loaded into the prompt before it has even started the task.

Tokens are only part of the cost. The scarce resource is the model's working set.

![Tool access competes for the same context budget](/assets/images/agent-harness/context-tool-budget-hero.png)

## Context Is an Execution Resource

Every tool has a context footprint.

The model needs to see the tool name, description, schema, usage rules, and sometimes examples. MCP servers make integration easier, but a large tool surface can quietly turn into a giant instruction bundle. Add project rules, memory files, prior messages, retrieved docs, logs, and raw tool results, and the agent starts every task carrying a backpack full of unrelated affordances.

This is a runtime constraint as much as a prompt hygiene issue.

| Context occupant | Hidden cost |
|---|---|
| Tool schemas | The model has to parse and compare more actions. |
| Rule files | Instructions compete with task-specific evidence. |
| Memory items | Old facts can distract from current state. |
| Raw logs | Verbose output can bury the useful failure signal. |
| Retrieved docs | Relevant snippets still compete for attention. |

Anthropic's [code execution with MCP post](https://www.anthropic.com/engineering/code-execution-with-mcp) describes the problem directly: tool definitions and intermediate results can overload the context window. Their proposed direction is to let the model use code execution as a way to interact with MCP tools more efficiently, keep large intermediate data out of the main attention stream, and retrieve only the pieces it needs.

The broader lesson is simple. The harness should manage tool attention the way an operating system manages resources. Load what is relevant. Keep bulky artifacts out of the hot path. Preserve full results somewhere else. Give the model enough routing information to discover the next tool without injecting every possible tool into every run.

## Tool Descriptions Are Part of the Program

Tool descriptions are easy to treat as documentation. In agent systems, they behave more like code.

They shape which action the model chooses. They define the difference between similar tools. They create hidden priorities. They can make a tool look safer, broader, narrower, cheaper, or more authoritative than it really is. A vague tool description is an ambiguous API. A bloated tool description is prompt debt.

This is why HumanLayer's [coding-agent harness essay](https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents) treats tool configuration as part of the harness. The model is not simply "using tools." It is interpreting a menu that the harness constructed. Too many MCP tools fill context. Poor tool descriptions create confusion. Hooks, rules, skills, and subagents become ways to shape what the model sees and when it sees it.

The important phrase is "when it sees it."

Most tasks need a narrow tool surface upfront. A coding agent might need repository navigation, shell commands, tests, and file edits for one task. It might need browser automation for another. It might need a database query for a third. Loading every capability into every run gives the model options it has no reason to consider.

That is how tool sprawl starts to look like dependency hell. The individual dependency may be useful. The combined surface becomes difficult to reason about.

## Progressive Disclosure Beats Tool Dumping

Good harnesses use progressive disclosure.

Instead of injecting every instruction and tool at the beginning, the harness gives the agent a small entry point and lets it discover deeper context when the task requires it. `AGENTS.md` can point to docs rather than contain the whole repo manual. Skills can load task-specific instructions only when triggered. Subagents can handle context-heavy exploration and return a compact result. Tool search can expose the right capability without placing every schema in the initial prompt.

OpenAI's [harness engineering post](https://openai.com/index/harness-engineering) gives a practical version of this idea. Their repository knowledge lived in docs, while `AGENTS.md` worked as a map. An agent-readable repo means the agent can find the right knowledge when needed, with a short entry point instead of a giant context dump.

HumanLayer's essay makes the same point from another angle with skills and subagents. Skills are useful because they disclose specialized instructions at the moment of need. Subagents are useful because they isolate expensive context and return distilled findings to the parent thread.

The design principle is to keep the main context window small enough for the model to think.

## Small Tool Surfaces Improve Control

Smaller tool surfaces also improve safety.

When a model has fewer actions available, permissions are easier to reason about. Approval gates become clearer. Logs become easier to inspect. Failures become easier to attribute. A human reviewer can understand which capabilities were in play during a run.

Agents can still be powerful. Capability should be routed through the harness instead of dumped into the prompt. A tool can exist without being visible. A capability can require discovery. A high-risk action can require a structured proposal, a dry run, or a human approval step before execution.

HumanLayer's [12 Factor Agents](https://www.humanlayer.dev/blog/12-factor-agents) frames tools as structured outputs. That framing is powerful because it separates the model's decision from the application's consequence. The model can propose an action. The harness decides what happens next.

That separation gets harder when the tool surface sprawls. Too many tools blur the action space. The model has to spend attention distinguishing verbs instead of solving the task. The developer has to debug behavior across a larger set of possible paths.

## A Practical Tool Budget

A harness should have a tool budget.

The budget is more than a count. It is a set of questions:

1. Does this tool need to be visible at task start?
2. Can the agent discover it later?
3. Can this tool be split into a safer proposal step and a deterministic execution step?
4. Does the tool return compact output by default?
5. Where do large results live outside the context window?
6. What approval boundary applies to this action?
7. What trace will explain why the model chose this tool?

| Add capability when... | Route capability when... |
|---|---|
| The task frequently needs it upfront. | It is useful only in specific branches. |
| The output is compact and actionable. | The output is large, noisy, or stateful. |
| The action is low risk. | The action needs approval or dry-run preview. |
| The tool has a clear unique role. | The tool overlaps with existing tools. |

These questions change how you add capabilities. A new tool adds an integration and a new path through the agent's decision space.

The same logic applies to memory. A memory item is useful only if it appears at the right time. A rule is useful only if it affects behavior. A log is useful only if the relevant part can be found. A retrieved document is useful only if it competes well against the other information in context.

The harness designer's job is to manage attention.

## The Agent Should Have Fewer Better Options

Tool sprawl is seductive because every added tool feels like an increase in possibility.

In real systems, possibility has to be shaped. The agent should have enough options to make progress and few enough options to choose well. It should see the tools relevant to the current task, the state needed to decide, and the evidence needed to verify the result. Everything else should be discoverable, stored, or handled by deterministic code outside the main loop.

That is the difference between a tool-rich agent and a tool-cluttered agent.

The first has access to capabilities through a harness that manages attention, permissions, output, and verification. The second starts every task with a crowded context window and hopes the model sorts it out.

Hope is a weak runtime strategy.

As agents get stronger, tool surfaces will get broader. That raises the value of harness discipline. Stronger models will use tools more effectively, but they will still operate through whatever context and action space the harness provides.

So before adding the next MCP server, memory layer, or tool group, ask what it will cost in attention.

Then decide whether the agent needs it now, later, or only through a narrower interface.
