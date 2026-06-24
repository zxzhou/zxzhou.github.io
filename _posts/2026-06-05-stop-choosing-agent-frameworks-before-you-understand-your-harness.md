---
layout: post
title: "Stop Choosing Agent Frameworks Before You Understand Your Harness"
date: 2026-06-05
tags: [ai-agents, agent-harness, llm, software-engineering]
excerpt: "Agent frameworks carry assumptions about context, state, control flow, tools, and verification. Harness thinking makes those assumptions visible before they become architecture."
---

Most agent framework debates start one layer too high.

By the time you are comparing graph runtimes, crew abstractions, SDK ergonomics, memory stores, tool registries, and deployment surfaces, you have usually accepted someone else's answer to the questions that matter most: what should the agent remember, who owns the loop, what enters the context window, which actions need permission, what counts as done, and how the system learns from failure.

Those are harness questions.

An agent harness is the system around the model that turns model output into repeatable work. It includes prompts and rule files, but it also includes context assembly, tools, state, permissions, sandboxes, evaluation loops, progress artifacts, observability, handoffs, and recovery logic. [LangChain's framing](https://www.langchain.com/blog/the-anatomy-of-an-agent-harness) is compact: "Agent = Model + Harness." [Pydantic's harness thesis](https://pydantic.dev/articles/the-harness-thesis) reaches the same conclusion from a production angle: stronger models still need durable state, observability, coordination, governance, and economic accountability around them before they become reliable agentic services.

This distinction matters because model comparisons are becoming a weaker explanation for why one agent system works and another one fails. Once the base model can plan, call tools, read errors, modify files, and recover from simple mistakes, the bottleneck moves into the surrounding system. The useful question shifts from "which model is smartest?" to "what environment lets this model do reliable work?"

That environment is the harness.

![Framework choice sits above harness boundaries](/assets/images/agent-harness/framework-harness-boundaries-hero.png)

## Frameworks Carry Theories Of Control

Agent frameworks carry theories of control.

LangGraph pushes you toward explicit graph state and node transitions. AutoGen pushes you toward multi-agent message passing. Crew-style frameworks push you toward role composition. Model-provider SDKs usually push you toward the execution loop their own agents were trained and tested inside.

Those opinions can be useful. They also answer architectural questions before your own system has produced evidence.

| Framework-first question | Harness-first question |
|---|---|
| Which framework should we use? | Which parts of the loop need to be owned by us? |
| Does it support memory? | What state must survive a run, and where does it live? |
| Does it support tools? | Which tools should enter attention, and when? |
| Does it support approvals? | Which actions need a human or policy gate before execution? |
| Does it support evals? | What evidence defines done for this task? |

The issue becomes sharper in agentic AI because the field is still moving. HumanLayer's [12 Factor Agents](https://www.humanlayer.dev/blog/12-factor-agents) makes the production-builder argument directly: many useful agent products are mostly deterministic software with LLM calls at specific decision points. The essay argues for owning prompts, context, control flow, and execution state instead of handing those surfaces to a black box.

Whether you adopt all twelve factors is secondary. The underlying move is the important part. Wrapping a model in a loop is easy. Deciding which parts of the loop should be deterministic, inspectable, interruptible, resumable, and domain-specific is the real design work.

When you choose the framework first, the framework becomes your architecture.

That can be fine in mature domains. In iOS development, UIKit and SwiftUI give you a stable vocabulary for views, layout, navigation, events, and state. The fundamentals have decades behind them. Agent systems lack that stability today. A framework's abstraction around memory, tool calling, subagents, approvals, or state replay may look reasonable this quarter and feel constraining after the next model release.

Anthropic's [harness design writeup](https://www.anthropic.com/engineering/harness-design-long-running-apps) captures the maintenance problem well. Every harness component encodes an assumption about where the model still needs help, and those assumptions can go stale as models improve. A framework-first approach can lock you into yesterday's answer to tomorrow's capability boundary.

## The Harness Owns The Surfaces That Matter

A useful harness makes four surfaces explicit.

First, it decides what state survives.

The context window is a working set. Durable memory has to live elsewhere. Long-running agents need files, git history, progress logs, execution plans, traces, database records, or other artifacts that survive context resets and session boundaries. Anthropic's [long-running agent harness](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) used an initializer agent, a progress file, git history, and structured handoffs so a fresh coding agent could understand what happened before it arrived. OpenAI's Codex team described a related pattern in its [harness engineering post](https://openai.com/index/harness-engineering): repository knowledge became the system of record, while `AGENTS.md` became a map into a larger `docs/` structure instead of a giant encyclopedia injected into every turn.

Second, the harness decides what enters attention.

Tool descriptions, rule files, long logs, MCP server schemas, memory files, and prior messages all compete for the same context budget. HumanLayer's [coding-agent harness essay](https://www.humanlayer.dev/blog/skill-issue-harness-engineering-for-coding-agents) is useful here because it treats context as a constrained resource rather than a dumping ground. Too many MCP tools inflate the system prompt. Too much raw test output distracts the agent. Subagents help because they isolate context-heavy work and return condensed results. Skills help because they disclose task-specific instructions only when needed. Anthropic makes a similar point in its [MCP code execution post](https://www.anthropic.com/engineering/code-execution-with-mcp): tool definitions and intermediate results can overload context, so the harness has to manage what the model sees.

Third, the harness decides where the model is allowed to act.

Tool calling is a structured proposal. Deterministic code still decides whether the proposed action runs now, waits for a human, pauses for an external event, retries with a compacted error, or stops. This is one of the strongest ideas in 12 Factor Agents: own your control flow. The model can choose a next step; the application controls the consequence.

Fourth, the harness decides how work is verified.

Agent reliability comes less from telling the model to be careful and more from giving it signals it can read. Tests, linters, screenshots, logs, traces, browser runs, evaluation suites, and review agents turn confidence into evidence. Anthropic's [agent evals guide](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) separates the agent harness, the eval suite, and the eval harness, but the practical lesson for builders is that evaluating an agent means evaluating the model and the surrounding system together. Datadog's [verification-loop post](https://www.datadoghq.com/blog/ai/harness-first-agents) pushes the same idea from a production angle: traces and telemetry make agent exploration cheaper because failures become visible and reusable.

These are architectural choices. A framework can help implement them. The discovery should happen one layer earlier.

## Start With The Smallest Useful Harness

There is a tempting failure mode in agent work: install a framework, connect a dozen tools, add memory, define roles, turn on observability, and hope robustness emerges from surface area.

Usually the opposite happens. The agent gets more options, more instructions, more state, and more ways to get distracted. The human gets less ability to understand which part of the system caused the behavior.

The better starting point is a small harness that exposes the right feedback loop.

For a coding agent, that might mean:

1. A short `AGENTS.md` that points the agent to deeper docs.
2. A deterministic test or lint command whose success output stays silent and whose failure output is concise.
3. A progress file or execution plan for work that crosses one session.
4. A small set of tools the agent actually needs.
5. A human approval gate for actions with irreversible effects.

Frameworks still have a place. The risk is using a framework to postpone thinking. After you understand the harness you need, a framework may become an implementation detail. Before that, it is a worldview adopted by accident.

OpenAI's Codex post is interesting for exactly this reason. The reported productivity gains came from a concrete operating environment rather than a generic framework story. The team made the repository legible to agents, built feedback loops into local development, used standard tools directly, versioned plans and decisions, and turned repeated failures into repository-level improvements. The important part is the role shift: engineers were designing environments, feedback loops, and control systems for agents to work inside.

That is harness engineering in the practical sense.

## Choose Frameworks After The Harness Is Legible

Once the harness is legible, framework selection becomes easier.

You can ask concrete questions instead of comparing marketing surfaces.

Does the framework let you own the context window, or does it hide prompt construction behind abstractions? Can you interrupt between tool selection and tool execution? Can you persist state in your own domain objects instead of an opaque run store? Can you route verbose tool output out of context and retrieve it later? Can humans and other systems resume an agent through simple APIs? Can production traces become eval cases or rule updates? Can you remove harness components when the model no longer needs them?

These questions change the evaluation. You are no longer asking whether LangGraph, AutoGen, CrewAI, the OpenAI Agents SDK, the Claude Agent SDK, or a homegrown loop is best. You are asking which one preserves the harness decisions you already understand.

Sometimes the answer will be a framework. Sometimes it will be a library plus a small loop. Sometimes it will be a model-provider agent because the model has been trained deeply inside that harness. Sometimes it will be a hybrid: provider harness for local coding work, your own orchestration for production workflows, external observability for traces, and plain files for durable handoff state.

The point is to make that choice after you know which assumptions are load-bearing.

Anthropic's long-running application harness is a good example of this discipline. It started with a planner, generator, evaluator, sprint contracts, Playwright-driven QA, and file-based communication. Later, as the model improved, some pieces became less load-bearing and the harness could be simplified. That is the right posture. Treat the harness as a living system whose complexity has to keep proving its value.

## The Real Artifact Is The Feedback Loop

Mitchell Hashimoto's [AI adoption post](https://mitchellh.com/writing/my-ai-adoption-journey) gives the most compact operational rule in this whole discussion: when an agent makes a mistake, engineer the environment so that mistake becomes less likely next time. Sometimes that means adding a line to `AGENTS.md`. Sometimes it means writing a script, a screenshot tool, a filtered test runner, or a validation check. The key is that the lesson enters the harness and becomes available to future runs.

This is where agent systems start compounding.

A prompt correction helps one run. A harness improvement changes the distribution of future runs. A test gives the agent a signal. A progress file lets another agent resume the work. A concise error hook lets the model correct itself without flooding the context window. A repository knowledge base lets the agent read the system instead of relying on a private chat transcript. An eval case turns a production failure into a reusable boundary.

The resulting system still depends on the model. Better models will make some harness pieces obsolete. They will also make more ambitious harnesses possible, because we will give agents longer tasks, higher stakes, broader tool access, and more responsibility. The boundary moves; the need for a surrounding system remains.

Framework debates are useful after this layer is visible. The durable skill is learning to see the harness: the state that survives, the context that enters attention, the actions that are allowed, the checks that define done, and the feedback that changes future behavior.

Choose the framework after that.

Until then, build the smallest harness that lets the agent fail in useful ways, then make every repeated failure harder to repeat.
