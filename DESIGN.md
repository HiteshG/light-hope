# Design Document

> Key decisions made while building the Light Agent Runtime.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      AgentRuntime                           │
│                                                             │
│   ┌────────────┐   ┌──────────────┐   ┌──────────────────┐ │
│   │RuntimeConfig│   │ ToolRegistry │   │ GuardrailEngine  │ │
│   │(all knobs)  │   │(schemas +    │   │(pre-execution    │ │
│   │             │   │ metadata)    │   │ safety checks)   │ │
│   └────────────┘   └──────┬───────┘   └────────┬─────────┘ │
│                           │                     │           │
│   ┌───────────────────────┴─────────────────────┘           │
│   │               Agent Loop                                │
│   │   ┌──────────────┐       ┌──────────────────┐          │
│   │   │ LLMProvider  │◄─────►│  ToolExecutor    │          │
│   │   │ (ABC)        │       │  (runs tools)    │          │
│   │   └──────┬───────┘       └────────┬─────────┘          │
│   │          │                        │                     │
│   │   ┌──────┴───────┐       ┌────────┴─────────┐          │
│   │   │MockProvider  │       │MockToolExecutor   │          │
│   │   │OpenAIProvider│       │(provided)         │          │
│   │   └──────────────┘       └──────────────────┘          │
│   └─────────────────────────────────────────────────────────│
│                           │                                 │
│                    ExecutionTrace                            │
│              (structured observability output)              │
└─────────────────────────────────────────────────────────────┘
```

**Data flow**: User request → `AgentRuntime.run()` → loop { LLM decides → GuardrailEngine checks → ToolExecutor acts → results fed back } → `ExecutionTrace`.

The runtime has **7 clearly separated components**:

| Component | File | Responsibility |
|-----------|------|---------------|
| **Types** | `types.py` | Shared data structures (Message, ToolCall, ToolResult, StepRecord, ExecutionTrace) |
| **LLM Provider** | `llm_provider.py` | ABC defining the provider contract |
| **Mock Provider** | `mock_provider.py` | Adapter wrapping MockLLMClient for testing |
| **OpenAI Provider** | `openai_provider.py` | Real GPT integration via the same ABC |
| **Tool Registry** | `tool_registry.py` | Tool schema + metadata management (separate from execution) |
| **Guardrail Engine** | `guardrails.py` | Pre-execution safety checks (3 policies) |
| **Config** | `config.py` | All tunable parameters in one dataclass |
| **Agent Runtime** | `agent.py` | Core orchestrator — the loop, error handling, trace building |

---

## Key Design Decisions

### 1. ABC for Provider Abstraction (not Protocol, not Duck Typing)

**Decision**: Use `abc.ABC` to define the `LLMProvider` interface.

**Alternatives considered**:
- **`typing.Protocol`** — structural typing, no inheritance required. Better for library code where you don't control the implementations.
- **Duck typing** — no interface at all, just match method signatures.

**Why ABC**: This is *platform infrastructure* that other engineering teams will extend. An explicit abstract base class:
- Makes the contract discoverable (`chat()` signature is documented)
- Surfaces errors at class definition time, not at runtime
- Supports `isinstance()` checks for defensive programming
- Clearly communicates "you MUST implement this" (vs. Protocol's implicit contract)

Protocol would've been acceptable; ABC is a deliberate choice for clarity in infrastructure code.

### 2. Separate ToolRegistry from ToolExecutor

**Decision**: Tool schemas/metadata live in `ToolRegistry`; tool execution lives in `MockToolExecutor`. They're different objects.

**Why not combine them?** The guardrail engine needs to ask "is `approve_invoice` mutating?" *before* executing it. If metadata lives inside the executor, the guardrail either (a) depends on the executor (tight coupling) or (b) can't check metadata at all. The registry is the single source of truth for tool properties; the executor is a dumb runner.

**Extensibility**: In production, the registry could load from a service catalog, while the executor makes HTTP calls to real APIs. They scale and change independently.

### 3. Guardrails in the Runtime, Not the Prompt

**Decision**: Safety checks are enforced deterministically by the `GuardrailEngine`, not via LLM prompts.

**Why**: The challenge explicitly states *"The runtime (not just the LLM prompt) shows awareness that bulk mutations require protection."* More fundamentally:
- LLM prompts are **advisory** — they can be ignored, jailbroken, or hallucinated past
- Runtime guardrails are **mandatory** — if the policy says "block," it blocks
- This is **defense-in-depth**: the prompt is layer 1, the runtime is layer 2

The mock LLM already "behaves well" in Scenario 4 (lists invoices and asks). But a real LLM might not. The runtime doesn't trust the LLM — it verifies.

### 4. Adapter Pattern for MockLLMClient

**Decision**: Created `MockProvider` as a thin wrapper around the provided `MockLLMClient`, rather than modifying the original.

**Why**: The challenge says you "should NOT need to modify" the provided files. The adapter pattern keeps the test fixture pristine while conforming it to our abstraction. This is also realistic — in production, you'd write adapters for third-party SDKs, not fork them.

### 5. Error Handling — Feed Errors Back to the LLM

**Decision**: Tool errors don't crash the runtime. They're wrapped in `ToolResult(status=ERROR)` and sent back to the LLM as conversation messages. The LLM then explains the error to the user.

**Why this over catching and returning immediately**: The LLM is better at user-facing communication than hardcoded error messages. In Scenario 3, the tool returns "Invoice not found: INV-9999" — the mock LLM wraps this into a helpful message. A real LLM would do even better.

**What the runtime catches**: every error. Tool exceptions → `ToolResult(ERROR)`. LLM exceptions → `trace.error`. Max iterations → `trace.error`. The runtime **never** raises to the caller; it always returns an `ExecutionTrace`.

### 6. Configurable Everything via RuntimeConfig

**Decision**: All behavioral parameters (max iterations, timeouts, guardrail thresholds, provider choice) live in a single `RuntimeConfig` dataclass.

**Why a dataclass**: Zero dependencies, type-safe, self-documenting with docstrings on each field. In production, this schema would be loaded from env vars or a config service — the dataclass is the *shape*, the loading is separate.

---

## Trade-offs & Limitations

1. **Synchronous execution**: The runtime is synchronous (no `async/await`). For the mock executor (5-50ms latency), this is fine. For real API calls with 100ms+ latency, you'd want async to overlap tool calls or handle timeouts with `asyncio.wait_for()`.

2. **Tool timeout not enforced**: `RuntimeConfig.tool_timeout_seconds` is defined but not actively enforced in the executor wrapper. The mock executor is fast enough that this doesn't matter, but production would need `signal.alarm()` or `asyncio.wait_for()`.

3. **No retry logic implemented**: `max_retries` and `retry_delay_seconds` are in the config but not wired into the agent loop. In production, retries should only apply to transient errors (timeouts, 5xx), not permanent ones (not found, invalid state).

4. **Guardrail for Scenario 4 depends on the mock LLM's behavior**: The mock LLM for Scenario 4 lists invoices first and then presents a confirmation message. It never actually tries to call `approve_invoice` multiple times. The bulk mutation guard is correctly wired but only triggers if the LLM actually requests multiple mutations — which the mock doesn't do. With a real LLM, the guard would be essential.

5. **No persistence**: Execution traces are in-memory only. Production would need durable storage (PostgreSQL, S3) for audit trails.

---

## What I'd Do Differently With More Time

1. **Async runtime**: Rewrite the agent loop with `asyncio` for concurrent tool execution, proper timeouts (`asyncio.wait_for()`), and non-blocking LLM calls. This is the #1 production requirement.

2. **Retry with exponential backoff**: Implement `tenacity`-style retries for transient tool/LLM failures with jitter: `delay = min(base * 2^attempt + random(0, jitter), max_delay)`.

3. **Streaming support**: Stream LLM responses token-by-token for lower perceived latency. The provider interface would add a `chat_stream()` method returning an async iterator.

4. **Tool-level permissions**: Check the current user's permissions against the tool's requirements before execution. E.g., `approve_invoice` requires `approve:invoices` permission.

5. **Structured logging with correlation IDs**: Each agent run gets a `trace_id`, propagated through all log entries. Use structured JSON logging for integration with observability platforms (Datadog, Grafana).

6. **Human-in-the-loop confirmation**: For Scenario 4, instead of just blocking, implement a real confirmation flow — pause the agent, surface the pending action to the user, resume on approval.
