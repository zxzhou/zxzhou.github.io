---
layout: post
title: "How I Organize My Workspace for AI Agents"
date: 2026-10-29
tags: [ai-agents, context-engineering, workflows]
excerpt: "A practical tour of my context infrastructure: why rules, workspace maps, skills, tools, and durable artifacts help an agent do useful work."
---

“Help me write a blog post” sounds like a complete request. In practice, it leaves a lot unsaid.

Who am I writing for? What have I already published? Which claims can I support with work I have actually done? What does my writing sound like? Where does the finished article belong?

An agent can start drafting immediately. To produce something that belongs on my website, though, it needs access to those answers.

I organize that information in a working directory alongside my projects. It contains instructions, a map of where things live, reusable task guides, tools, and saved work. I call this my **context infrastructure**: the environment that helps an agent discover what it needs to carry out a task.

My setup draws on [grapeot’s context-infrastructure reference implementation](https://github.com/grapeot/context-infrastructure). The useful part of adopting it has been deciding which pieces fit my own work, adapting the instructions, and keeping the active system small enough to understand. Here is how those pieces fit together.

## Start with a task the reader can follow

This article provides a concrete example. When I asked for help writing about my setup, the agent first read the workspace rules and communication preferences. It followed the workspace map to my website repository, inspected existing posts, and consulted the web-search guide before checking the live homepage.

Those steps changed the writing problem. The agent could see that I already had articles about agent harnesses and handoffs. It could therefore suggest a practical walkthrough that complemented those posts. It also found an important limit in the workspace instructions: automatic memory accumulation is currently disabled.

That detail matters. The reference project includes a broader memory architecture. Describing every available component as part of my active workflow would have overstated what I use.

This is the kind of assistance I want: an agent that investigates the environment, distinguishes current configuration from possible extensions, and uses what it finds to make better decisions.

## The components answer different questions

The directory structure is ordinary. The responsibilities assigned to it are what make it useful.

```text
workspace/
├── AGENTS.md                  # Where the agent starts
├── rules/
│   ├── SOUL.md                # General working principles
│   ├── USER.md                # Relevant background and preferences
│   ├── COMMUNICATION.md       # Writing and communication conventions
│   ├── WORKSPACE.md           # Where projects and records live
│   └── skills/
│       ├── INDEX.md          # Available task guides
│       └── ...
├── contexts/                  # Saved research and working material
├── tools/                     # Reusable executable utilities
├── adhoc_jobs/                # Individual projects and experiments
└── periodic_jobs/             # Optional scheduled workflows
```

These names are conventions in my workspace. An agent needs an instruction or integration that leads it to them. A Markdown file sitting on disk only helps once the agent discovers and reads it.

| Component | Question it answers | Why it helps |
|---|---|---|
| Entry instructions | What should I read first? | Gives a new session a consistent starting point |
| User and communication context | What makes this result useful to this person? | Reduces repeated explanation of preferences |
| Workspace map | Where should I look and save work? | Narrows searches to relevant locations |
| Skills | How should I approach this kind of task? | Preserves reusable methods and tool guidance |
| Tools | How can I act and inspect the result? | Connects instructions to executable operations |
| Saved artifacts | What has already happened? | Supports review and continuation across sessions |

## Entry instructions establish a starting point

My `AGENTS.md` tells the agent to read the core rules before working, consult the workspace map before searching, and look for an applicable skill before inventing a process.

This provides continuity across sessions. I can start with the task itself because the workspace supplies a standard orientation procedure.

The entry file also establishes boundaries, including how to handle private information and destructive actions. Written instructions communicate intent; actual access controls and tool permissions still determine what an agent can do.

I want this entry point to remain short enough that its priorities are visible. Detailed instructions belong near the work they govern. Putting every workflow into the first file would make routine tasks carry the context of unrelated projects.

## Personal context defines what a good result looks like

A technically correct answer can still require substantial editing before I can use it.

For writing, I care about precise claims, concrete examples, and a restrained tone. For technical work, relevant background helps the agent judge how much explanation I need. These preferences live in `USER.md` and `COMMUNICATION.md`, while `SOUL.md` describes broader working principles, such as investigating available information before asking me to provide it again.

Separating these files gives each kind of instruction a place. A change in writing style should be easy to find without searching through deployment instructions or research notes.

The useful content is specific enough to affect a decision. “Write clearly” leaves substantial room for interpretation. “State the concrete mechanism and avoid promotional adjectives” gives the agent something it can apply while editing a paragraph.

Personal context also requires curation. A copied profile can describe someone else convincingly. I treat inherited templates as material to review and adapt, and keep sensitive details out of public examples.

## A workspace map reduces unnecessary searching

My workspace contains code, research, blog posts, and experiments. Without a map, finding the right material can require a broad search with many irrelevant results.

`WORKSPACE.md` provides a compact routing table. It tells the agent that the website lives in `zxzhou.github.io/`, individual projects belong under `adhoc_jobs/`, and research reports live under `contexts/survey_sessions/`.

For this article, that was enough to locate the existing blog without asking me for its path.

The map is deliberately less detailed than the directories themselves. It points to the relevant area; the agent then inspects actual files. This keeps navigation lightweight while leaving the repository as the source for current implementation details.

The tradeoff is maintenance. An outdated map can send the agent confidently to the wrong location. When a project moves, its routing entry needs to move with it.

## Skills preserve methods I want to reuse

Some tasks recur often enough that their working method deserves a permanent home.

A skill in this setup is a task guide. It can describe when to use a capability, which tools are available, what inputs they require, and how to check the output. The skills index provides a brief overview; the agent reads the detailed guide when the task calls for it.

The web-search skill is a concrete example. It points to the local Tavily command-line interface and documents how to search or extract a page. For this article, the agent used that guide to retrieve the homepage rather than guessing how the installed search utility worked.

This arrangement separates discovery from execution. The index answers “what capabilities are available?” The individual guide answers “how do I use this one here?” Loading details only when needed keeps unrelated procedures out of the current task.

A useful skill should also leave room for judgment. A research task needs a question, source standards, and a deliverable. The agent should still be able to follow a useful lead or investigate missing information. I want repeatable quality with enough flexibility to handle the actual problem.

## Tools make the workflow executable

Instructions become more useful when the agent can act on them.

A tool might retrieve a webpage, run a build, inspect a file, or transform data. Its interface matters because the agent needs to understand both what happened and what to do next. Structured output, explicit errors, and saved result paths make that easier.

A skill and a tool serve different purposes. The skill explains when and how to use a capability. The tool performs an operation. The surrounding task determines whether that operation produced the intended result.

For a blog post, writing Markdown completes only part of the job. The site still needs to build, the post needs a working URL, and the homepage needs to lead readers to it. Those checks connect the writing task to the website a reader will actually use.

I judge an integration by whether it closes that gap between generating an output and confirming that the output works in its destination.

## Saved artifacts let the next session continue

Rules describe recurring expectations. Project artifacts describe a particular piece of work.

A research report can preserve sources and unresolved questions. A code change can preserve implementation decisions. A short handoff note can record what was completed, what was checked, and what remains.

This gives a new session something concrete to inspect. It can compare a note with the files and verification results instead of relying entirely on a previous conversation's summary.

I discussed this more broadly in [my article about long-running agents and handoffs](https://zxzhou.github.io/blogs/2026/06/12/long-running-agents-fail-at-handoffs-before-they-fail-at-reasoning/). In a personal workspace, the practical version can be small: save the deliverable in the expected location and leave enough state for the next task to begin without reconstructing everything.

For longer work, I would include the goal, changed files, verification results, open issues, and next action. These records need updating as work progresses. A stale handoff can mislead an agent just as easily as a stale directory map.

## Automatic memory and scheduling are optional extensions

The reference architecture includes mechanisms for accumulating observations, reflecting on them, and running scheduled workflows. My current workspace instructions keep automatic memory accumulation disabled and use the persistent rules layer as the active baseline.

That is an intentional scope choice. Automatic memory raises additional questions: which observations deserve to persist, how mistakes get corrected, when old information expires, and which private details should never enter a shared artifact.

Scheduling introduces a separate responsibility. A recurring job needs a defined input, a useful output, a way to detect failure, and a policy for notifying the user. Running it repeatedly does not establish those properties.

I would add these components for a specific recurring need, with checks appropriate to that workflow. The simpler setup already supports orientation, tool use, and saved work without requiring an autonomous background process.

## How I would check whether the setup helps

The relevant question is whether the environment changes observable behavior.

For a writing task, I can check whether the agent finds existing posts, follows the requested style, supports factual claims, saves the article correctly, and verifies the generated site. For a coding task, I can check whether it finds the right project, respects local conventions, and runs checks relevant to the change.

A useful experiment would give the same bounded task to fresh sessions with and without the workspace guidance, while keeping the model and tool access comparable. I would track corrections required, irrelevant exploration, and successful completion. Repeating the comparison would be more informative than choosing one impressive run.

I have not established a controlled productivity improvement for this setup. The example here demonstrates how the components guide a task; it does not establish a percentage gain in speed or quality.

The practical habit I want to retain is to improve the environment when a recurring failure has a clear cause. If the agent searches the wrong directory, fix the map. If it repeatedly misses a writing requirement, clarify the relevant rule. If a result looks plausible but fails in use, add a check that exposes the failure.

Each addition should earn its place by helping with actual work. A small workspace that an agent can navigate, act within, and leave in an understandable state is a useful starting point.
