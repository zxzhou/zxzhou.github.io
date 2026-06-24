---
layout: post
title: "Long-Running Agents Fail at Handoffs Before They Fail at Reasoning"
date: 2026-06-12
tags: [ai-agents, agent-harness, context-engineering, llm]
excerpt: "The hard part of long-running agents is turning transient reasoning into durable state. A useful harness makes work resumable across context windows, sessions, tools, and reviewers."
---

Long-running agents usually look impressive for the first few minutes.

They read the task. They inspect the repo. They form a plan. They make a few edits. They run tests. The demo makes autonomy feel close.

Then the work crosses a boundary.

The context window fills. A new session starts. A subtask moves to another tool. A human reviews halfway through. A branch diverges. A build fails after the original agent has already moved on. At that point, the problem is often described as a reasoning failure. The agent "forgot" the plan, repeated work, chased the wrong error, or optimized for the wrong success condition.

That diagnosis is too convenient. The deeper issue is usually a handoff failure.

A context window is a working set. It is where the model thinks right now. Durable memory has to live somewhere else. Long-running agents need a harness that converts transient reasoning into state other workers can inspect, resume, and verify.

![Long-running work survives through durable artifacts](/assets/images/agent-harness/handoff-memory-system-hero.png)

## The Context Window Is Not a Project Memory

A single context window can hold a lot of text, but it is still a poor project memory.

It is expensive to fill. It is easy to pollute. It mixes facts, guesses, tool output, old mistakes, reviewer comments, and implementation details into one attention stream. Summarization helps, but a summary is still a lossy transformation controlled by whatever the model thought mattered at that moment.

This matters more as tasks get longer. A model can reason well inside a single local problem and still perform badly across a multi-hour project. The failure is not always that it lacks intelligence. The failure is that the system did not preserve the right intermediate artifacts.

Anthropic's post on [effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) is useful because it treats continuity as an engineering problem. Their harness used an initializer agent, progress files, git history, setup scripts, and structured handoff artifacts. A new coding agent could arrive later and understand the state of the work without depending on the previous chat transcript.

That is the core move. The transcript should not be the system of record.

| Weak memory surface | Stronger harness artifact |
|---|---|
| Chat transcript | Progress file with current state and next action |
| "I think tests pass" | Test command and recorded result |
| Model summary | Commit history and concrete diff |
| Pasted logs | Stored logs with concise failure excerpts |
| Human recollection | Handoff note readable by the next worker |

OpenAI's Codex team describes a related pattern in its [harness engineering post](https://openai.com/index/harness-engineering). `AGENTS.md` became a table of contents into deeper repository knowledge. The durable source of truth lived in files, docs, tests, and repo structure. The agent could re-enter the work by reading the environment rather than relying on a private conversation.

## Handoffs Are the Unit of Autonomy

A long-running agent system should be judged by how well it handles handoffs.

Can a fresh agent resume the work after context reset? Can a human understand what happened without reading the whole transcript? Can another tool pick up the next step? Can a failure become visible to the next run? Can the system tell the difference between work that is done, work that is blocked, and work that merely looks complete?

Those questions are more revealing than "how long can the model think?"

Human teams learned this a long time ago. Real projects survive because important state leaves people's heads. We write specs, tickets, commit messages, tests, design docs, incident reports, runbooks, and review comments. These artifacts are not bureaucracy when they are well designed. They are how work survives interruption.

Agents need the same thing, just with tighter feedback loops.

For coding agents, durable state usually has a few practical forms:

1. A plan file that names the goal, constraints, current decision, and remaining work.
2. Git commits or diffs that make the actual state of the code explicit.
3. Test and lint commands that encode the current definition of correct.
4. A progress log that records blockers, failed attempts, and open questions.
5. A handoff note that tells the next agent where to resume.

The exact format matters less than the property. The next worker should reconstruct the current state from artifacts, not from vibes.

## Long Tasks Need Shift Changes

The phrase "long-running agent" can create the wrong picture. It suggests one agent staying coherent for hours.

A better picture is shift work.

One agent initializes the task. Another implements a bounded slice. Another evaluates. Another resumes after a failure. A human enters at approval points. The harness coordinates these transitions and keeps the state legible.

Anthropic's later [harness design writeup](https://www.anthropic.com/engineering/harness-design-long-running-apps) makes this explicit with planner, generator, and evaluator roles. The important detail is not the exact three-agent architecture. The important detail is that each role produces artifacts the next role can inspect. Sprint contracts, QA results, screenshots, and files become coordination media.

That design has a cost. More handoffs mean more orchestration, more artifacts, more token use, and more places for mismatch. The trade-off becomes reasonable when the task is large enough that a single context thread loses coherence or when the verification burden is too high for one agent to grade its own work.

This is why harness design has to stay empirical. Some tasks need a multi-agent relay. Some tasks need one small loop and a test runner. Some tasks need a human gate. The harness should grow from observed failure modes, not from an abstract desire to make the architecture look agentic.

## Resumability Is the First Real Test

A useful test for any agent harness is simple: stop it in the middle and restart with a fresh agent.

If the next agent can reconstruct the goal, current state, constraints, completed work, failed attempts, and next action, the harness has real memory. If the next agent has to infer all of that from a pasted transcript, the system is still a chat session with extra tools.

This test also exposes weak documentation. `AGENTS.md` cannot become a giant rules dump. OpenAI's pattern is better: keep the entry point short and point to deeper docs. The same principle applies to progress files. A good handoff is compact enough to read and specific enough to act on.

The handoff should answer practical questions:

1. What is the goal?
2. What changed?
3. What has been verified?
4. What failed?
5. What should happen next?
6. Which action needs human approval?

Those questions sound basic because reliable systems are often basic at the artifact layer. The sophistication comes from putting the right artifact in the right place, then making the agent actually use it.

| Handoff field | Why it matters |
|---|---|
| Goal | Prevents the next worker from optimizing a stale target. |
| Changed files | Anchors the handoff in repo reality. |
| Verification | Separates completed work from plausible work. |
| Failed attempts | Prevents repeated loops through the same dead end. |
| Next action | Makes resumption cheap. |
| Approval needed | Keeps irreversible actions outside autonomous drift. |

## The Harness Is the Memory System

Mitchell Hashimoto's [AI adoption post](https://mitchellh.com/writing/my-ai-adoption-journey) gives a useful operational habit: when an agent makes a mistake, improve the environment so that mistake is less likely next time.

For long-running work, that usually means improving memory. Add a rule only if the agent will find it at the right time. Add a script if the mistake can be detected. Add a progress artifact if work gets lost between sessions. Add a test if correctness can be checked. Add a handoff format if different agents or humans keep reconstructing the same context.

The point is to move lessons from human memory into the harness.

That is how agent systems compound. A better prompt helps one run. A better handoff format helps every resumed run. A better test helps every future agent see the same boundary. A better repo map helps agents navigate without filling the context window with stale explanations.

Long-running agents will keep improving as models get stronger. The model boundary will move. The need for durable state will remain, because real work crosses time, tools, people, and failure modes.

So the first question for a long-running agent is not "can it reason for hours?"

The better question is: can the work survive a handoff?
