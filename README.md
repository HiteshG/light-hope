# Light AI Platform Engineer — Take-Home Challenge

## The Challenge

Build a **mini agent runtime** that orchestrates an LLM-powered assistant for Light's finance platform.

You are not building a chatbot. You are building **infrastructure** — the engine that sits between a user request and a set of tools, using an LLM to decide what to do, executing actions, handling failures, and producing a clear record of everything that happened.

This is the kind of system the AI Platform team at Light is building for real: a runtime that product engineers across the company will use to ship AI features.

---

## Context

Light is building an AI assistant that helps finance teams manage invoices. The assistant has access to a set of **tools** — functions it can call to query data, approve payments, and send notifications.

Your job is to build the **runtime** that powers this assistant:

```
User request
    ↓
[ Your Agent Runtime ]
    ↓↑              ↓↑
   LLM           Tool Executor
 (decides)        (acts)
    ↓
Structured execution trace + final response
```

We've provided:
- **Tool definitions** (`data/tools.json`) — JSON schemas describing 5 available tools
- **Mock data** (`data/mock_data.json`) — the invoices and user data the tools operate on
- **Mock tool executor** — a working implementation that simulates the Light API
- **Mock LLM client** — scripted responses for testing without an API key
- **Test scenarios** (`data/scenarios.json`) — 5 scenarios graded by difficulty

---

## What We're Looking For

This is a **platform engineering** challenge, not an AI/ML challenge. We're evaluating:

| Signal | What we're looking at |
|--------|----------------------|
| **Systems design** | How you structure the runtime — abstractions, separation of concerns, extensibility |
| **Reliability thinking** | How you handle errors, timeouts, retries, and edge cases |
| **Observability instinct** | Whether the execution trace tells a clear story of what happened and why |
| **LLM provider abstraction** | Whether the runtime is decoupled from a specific LLM provider |
| **Safety awareness** | How you handle mutating operations, bulk actions, and guardrails |
| **Testing approach** | How you verify the runtime behaves correctly across scenarios |
| **Communication** | How clearly you explain your decisions in DESIGN.md and PRODUCTION.md |

We are **not** evaluating prompt engineering skill, UI/UX, or how pretty the output looks.

---

## Requirements

### Must Have (scenarios 1–3 should pass)

1. **Agent loop**: An LLM ↔ tool execution loop that runs until the LLM produces a final answer
2. **Tool execution**: The runtime calls tools based on what the LLM requests and feeds results back
3. **Error handling**: Tool errors (not found, invalid state, etc.) are handled gracefully — the agent doesn't crash
4. **Execution trace**: Every run produces a structured trace showing the steps taken, tools called, results received, and timing
5. **Provider abstraction**: The LLM client is behind an interface/protocol that could be swapped (mock ↔ OpenAI ↔ Anthropic) via configuration, not code changes

### Nice to Have (scenarios 4–5, and beyond)

6. **Guardrails for mutations**: The runtime is aware that some tools are dangerous (mutating, irreversible) and applies safeguards
7. **Configurable behavior**: Timeouts, max iterations, retry policies, etc. are configurable — not hardcoded
8. **Real LLM integration**: Wire up a real provider (OpenAI, Anthropic, etc.) behind your abstraction, in addition to the mock
9. **Robust test suite**: Tests that verify not just outcomes but runtime behavior (ordering, error propagation, guardrail enforcement)

---

## Test Scenarios

These are defined in `data/scenarios.json`. Here's the summary:

| # | Name | Difficulty | User Request | Key Test |
|---|------|-----------|--------------|----------|
| 1 | Simple Query | Basic | "Show me all unpaid invoices over €5,000" | Basic loop works |
| 2 | Multi-Step | Basic | "Find the largest pending invoice from Acme Corp and approve it" | Multi-tool chaining |
| 3 | Error Handling | Basic | "Approve invoice INV-9999" | Graceful failure |
| 4 | Bulk Mutation | Advanced | "Approve all pending invoices" | Safety guardrails |
| 5 | Ambiguity | Advanced | "What's the status of the Globex invoice?" | Handling multiple matches |

**Expectations:** Scenarios 1–3 should pass. Scenarios 4–5 are stretch goals that differentiate strong submissions.

---

## The Mock LLM

We provide a `MockLLMClient` that returns scripted responses for each scenario. This lets you build and test your runtime without needing an API key.

**However:** your runtime should interact with the LLM through an abstraction (interface, protocol, base class, etc.) — not directly coupled to the mock. We encourage you to also wire up a real LLM provider if you have access to one. If you do, include brief setup instructions.

The mock is a convenience for development, not a constraint on your design.

---

## Deliverables

Please submit:

| File | What | Guidance |
|------|------|----------|
| **Your code** | The agent runtime implementation | Use any language. Python starter provided in `starter/` — use it, adapt it, or start fresh. |
| **DESIGN.md** | Key design decisions you made | Template in `templates/DESIGN.md`. ~1–2 pages. |
| **PRODUCTION.md** | How you'd take this to production | Template in `templates/PRODUCTION.md`. ~1–2 pages. |
| **Passing tests** | Scenarios 1–3 at minimum | Adapt the stubs in `starter/tests/test_scenarios.py` or write your own. |

---

## Getting Started (Python starter)

```bash
cd starter/

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Run the skeleton (will show "not yet implemented")
python -m light_agent.runner

# Run tests (will show failures until you implement)
pytest tests/ -v
```

**Using a different language?** Grab the data files from `data/` (tool schemas, mock data, scenarios) and build from scratch. The JSON files are the contract — everything else is a convenience.

---

## Time Expectation

We expect this to take **2–4 hours**. We would rather see a well-designed partial solution with a thoughtful DESIGN.md than a complete but messy implementation. Show us how you think, not just how you code.

If you find yourself going past 4 hours, stop and document what you'd do next in your DESIGN.md. Knowing when to stop and communicate is a skill we value.

---

## What Happens Next

After you submit, we'll schedule a **technical interview** (~60 min) where we'll:

1. Walk through your code and design decisions together
2. Discuss your PRODUCTION.md and dive deeper into specific topics
3. Explore extensions: "How would you add X?" / "What if Y happened?"
4. Branch into a broader technical conversation

This isn't a gotcha — it's a conversation. We want to understand how you think about building systems, not quiz you on trivia.

---

## Questions?

If anything is unclear, please reach out. We'd rather you ask than guess. Ambiguity in a challenge brief is a bug, not a test.

Good luck — we're excited to see what you build.
