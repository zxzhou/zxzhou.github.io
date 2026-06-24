---
layout: post
title: "Agents Need Stop Conditions More Than Pep Talks"
date: 2026-06-19
tags: [ai-agents, agent-harness, evals, observability]
excerpt: "Telling an agent to be careful is weak control. A harness gives the model external signals: tests, traces, screenshots, evals, policy gates, and human approvals."
---

"Be careful" is one of the weakest controls in an agent system.

It sounds responsible. It feels like engineering discipline. In practice, it usually means the harness has failed to provide a better signal.

An agent that can call tools, edit files, browse pages, or trigger workflows needs a stronger definition of done than "the model says it is done." It needs checks it can read, traces humans can inspect, and stop conditions the system can enforce. Otherwise every run ends in the same uncomfortable place: a fluent answer, a confident summary, and an unclear relationship to reality.

The next stage of agent engineering is less about writing better motivational prompts and more about designing verification surfaces.

![Online traces and offline evals feed agent stop conditions](/assets/images/agent-harness/verification-loops-hero.png)

## Confidence Is Not Evidence

LLMs are good at producing plausible explanations of their own behavior. That makes agent work dangerous in a specific way. The system can sound finished before the work is finished.

This is easy to see in coding. An agent may claim that tests pass after running the wrong command. It may fix the visible error and leave the real bug intact. It may produce a polished UI that fails when clicked. It may summarize a plan as completed because the local transcript points in that direction, while the repository state says otherwise.

The harness has to create external evidence.

| Weak control | Stronger harness signal |
|---|---|
| "Be careful" | Test, lint, or policy gate |
| "Check your work" | Independent evaluator or review agent |
| "Only do safe actions" | Permission boundary before execution |
| "Stop when done" | Explicit done criteria tied to evidence |
| "Try again" | Retry limit with escalation path |

Tests are evidence. Linters are evidence. Browser screenshots are evidence. Tool traces are evidence. Diff reviews are evidence. Human approvals are evidence. Evals are evidence when they represent a real failure boundary. Logs and telemetry are evidence when they connect the agent's decisions to actual system behavior.

The agent should read those signals, and the human should be able to audit them later.

This is why [Anthropic's guide to agent evals](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) separates three things that often get mixed together: the agent harness, the eval suite, and the eval harness. That distinction is practical. The agent harness runs the system. The eval suite defines what behavior you care about. The eval harness executes those cases and measures results. If those layers blur, you end up testing the wrong thing or trusting the wrong signal.

## Stop Conditions Are Product Decisions

Every agent loop needs an answer to a basic question: when should this stop?

There are several possible answers. Stop when tests pass. Stop when the evaluator approves. Stop when the user confirms. Stop when the agent has attempted the same tool three times. Stop when the diff exceeds a certain size. Stop when a policy boundary is reached. Stop when confidence is low and the next action is irreversible.

Those are product decisions as much as engineering details.

A customer-support agent should stop before refunding a large order without approval. A coding agent should stop before rewriting unrelated files. A data agent should stop before sending a report whose inputs failed validation. A research agent should stop when sources disagree on a claim that drives the conclusion.

The harness owns this boundary. The model can propose that the work is complete, blocked, risky, or ready for review. Deterministic code decides what that status permits.

HumanLayer's [12 Factor Agents](https://www.humanlayer.dev/blog/12-factor-agents) is strong on this point because it treats tool calls as structured outputs and control flow as application logic. The model may output an intent. Your system decides whether that intent runs synchronously, waits for a human, pauses, retries, or exits. That design lets stop conditions exist outside the model's self-assessment.

## Verification Has Two Loops

A useful harness has at least two verification loops.

The first loop is offline. You collect known failure modes and turn them into evals. This is where regression discipline enters the agent system. If an agent once hallucinated a policy, broke a workflow, ignored a permission boundary, or failed a task-specific requirement, the harness should make that failure easier to detect next time.

The second loop is online. You observe real runs and find failure modes you did not know to test. This is where traces matter. An agent trace can show which context entered the model, which tools were offered, which tool calls were attempted, what failed, what was retried, what was hidden from the model, and where a human intervened.

| Loop | Input | Output |
|---|---|---|
| Offline evals | Known failure modes | Regression cases and score thresholds |
| Online traces | Real agent runs | New failure modes and debugging evidence |
| Stop conditions | Policy, risk, and verification signals | Pause, retry, escalate, or complete |

Datadog's [harness-first agents post](https://www.datadoghq.com/blog/ai/harness-first-agents) argues for observability-driven verification because agent behavior is exploratory. No team enumerates every path upfront. Production traces become a source of future tests, rules, dashboards, and guardrails.

That feedback loop is the real asset. A single eval suite can go stale. A single dashboard can become noise. A harness that converts observed failures into sharper stop conditions keeps learning.

## The Agent Should See the Consequences

Verification works best when the agent can read the consequence of its action.

For code, that means tests, type checks, lint output, screenshots, logs, and runtime behavior. For data work, that means schema validation, row counts, source provenance, and sanity checks. For external actions, that means dry-run previews, approval prompts, idempotency checks, and audit logs.

The model needs selected evidence in context. Too much raw output can make the system worse. The harness should route large logs and traces to storage, then feed back the part that matters: the failing assertion, the relevant screenshot, the policy violation, the exact command output, or the concise diff summary.

This is a design problem. A bad harness floods the model with evidence until it loses the thread. A good harness compresses evidence into a signal the model can act on while preserving the full artifact for human inspection.

Anthropic's [harness design post](https://www.anthropic.com/engineering/harness-design-long-running-apps) shows this pattern in a concrete setting. A generator agent builds. An evaluator agent tests the app through Playwright and produces feedback. The system does not rely on the generator's self-review as the final authority. It creates a separate evaluation surface.

That separation is one of the most important ideas in agent reliability. The worker and the judge can both use language models, but they should not be the same loop with the same incentives and the same blind spots.

## Repeated Failures Should Become Boundaries

The practical rule is straightforward: every repeated failure should become a boundary in the harness.

If the agent keeps editing unrelated files, narrow the working set or add diff checks. If it keeps stopping after superficial success, add a verification checklist tied to real commands. If it keeps retrying the same broken tool call, add a retry limit and escalation path. If it keeps misreading source material, add citation requirements and source extraction checks. If it keeps producing UI that looks plausible but fails on interaction, add browser tests or screenshot review.

This is how reliability compounds. The prompt can remind the agent to be careful. The harness can make care observable.

The difference matters. A reminder depends on the model preserving the instruction, interpreting it correctly, and applying it under pressure. A stop condition changes what the system allows. A test changes what counts as done. A trace changes what humans can debug. An eval changes whether a failure stays anecdotal or becomes reusable.

Agent systems become trustworthy when confidence has to pass through evidence.

So the next time an agent makes a mistake, the best fix may not be another paragraph in the system prompt. The better question is which signal was missing.

Then put that signal into the harness.
