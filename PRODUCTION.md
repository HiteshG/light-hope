# Production Readiness Plan

> How I'd take this agent runtime to production at Light, serving multiple product teams building AI features for finance customers.

## Deployment & Operations

### Architecture

Deploy as a **stateless microservice** behind an API gateway. The runtime holds no state between requests — all state lives in the execution trace returned to the caller.

```
Client → API Gateway → Agent Runtime Service → LLM Provider (OpenAI/Anthropic)
                                             → Tool Service (Light API)
                                             → Trace Store (PostgreSQL/S3)
```

### Infrastructure

- **Container**: Docker image with the Python runtime, deployed on Kubernetes (EKS/GKE)
- **Scaling**: Horizontal pod autoscaler based on request queue depth (not CPU — the service is I/O-bound waiting on LLM calls)
- **Config**: Environment variables for secrets (API keys), feature flags (LaunchDarkly) for runtime config like guardrail thresholds and model selection
- **Load balancing**: L7 load balancer with health checks on `/healthz` (simple) and `/readyz` (checks LLM connectivity)

### Rollouts & Rollbacks

- **Canary deployments**: Route 5% of traffic to the new version, monitor error rates and latency for 15 minutes before full rollout
- **Rollback**: Kubernetes rollback to previous ReplicaSet. The runtime is stateless, so rollbacks are instant — no data migrations
- **Version pinning**: Each deployment pins the LLM model version (e.g., `gpt-4o-mini-2024-07-18`, not `gpt-4o-mini`). Model version changes are treated as deployments, not transparent upgrades

**Why version-pin models?** A "minor" model update can change tool-calling behavior, introduce regressions in reasoning, or alter output formatting. Treating model changes as deployments gives us the same rollback/canary guarantees we have for code changes.

---

## Observability & Monitoring

### Execution Traces

Every agent run already produces a structured `ExecutionTrace`. In production, persist these to a trace store:

```json
{
  "trace_id": "abc-123",
  "user_request": "Approve all pending invoices",
  "steps": [...],
  "tool_calls_made": 1,
  "llm_calls_made": 2,
  "total_duration_ms": 1250,
  "guardrails_triggered": [{"policy": "bulk_mutation_guard", ...}],
  "model": "gpt-4o-mini-2024-07-18",
  "team": "finance-ops",
  "timestamp": "2026-03-01T10:30:00Z"
}
```

### Key Metrics (Datadog / Prometheus)

| Metric | Type | Alert Threshold |
|--------|------|----------------|
| `agent.request.duration_ms` | Histogram | P95 > 10s |
| `agent.request.error_rate` | Counter | > 5% over 5 min |
| `agent.llm.calls_per_request` | Histogram | P99 > 8 |
| `agent.tool.error_rate` | Counter by tool | > 10% for any tool |
| `agent.guardrail.triggered` | Counter by policy | — (informational) |
| `agent.max_iterations_hit` | Counter | > 1% of requests |
| `agent.llm.latency_ms` | Histogram by provider | P95 > 5s |
| `agent.cost.tokens_used` | Counter by model | — (for billing) |

### Detecting Misbehavior

- **Hallucination detection**: Compare tool names requested by the LLM against the registry. Log `unknown_tool` events. Alert if rate exceeds 2% — suggests a model degradation.
- **Loop detection**: Alert on `max_iterations_hit`. Consistently hitting the limit means the agent is stuck in a reasoning loop.
- **Guardrail frequency**: Dashboard showing guardrail triggers by policy. A spike in `bulk_mutation_guard` might mean users are discovering a new workflow that needs proper support.
- **Regression testing**: Run the 5 test scenarios against each new model version. Compare execution traces against golden traces — same tools called, same order, same assertions pass.

---

## Safety & Governance

### Defense in Depth (3 Layers)

```
Layer 1: LLM System Prompt    → Advisory ("don't approve without checking")
Layer 2: Runtime Guardrails   → Mandatory (GuardrailEngine blocks bulk mutations)
Layer 3: Backend Permissions   → Enforcement (API checks user.permissions before executing)
```

Each layer operates independently. Even if Layer 1 fails (LLM ignores the prompt), Layer 2 blocks the action. Even if Layers 1+2 both fail (bug in guardrail code), Layer 3 enforces permissions at the API level.

### Audit Trail

Every mutating action is recorded in the execution trace with:
- Who requested it (user identity from auth)
- What the LLM decided (full conversation history)
- What the guardrail engine said (allow/block/flag + reason)
- What the tool returned (success/error)
- Timestamps for every step

This gives compliance teams a complete chain of accountability: **user → request → LLM reasoning → guardrail decision → action → outcome**.

### Guardrail Policies (Configurable Per Team)

| Policy | Default | Override Example |
|--------|---------|-----------------|
| `max_mutations_without_confirmation` | 1 | Data ops team: 10 (they need batch operations) |
| `enable_guardrails` | true | Internal dev tools: false (trusted environment) |
| `tool_timeout_seconds` | 30s | Long-running reports: 120s |

Policies are per-team, not global. Finance teams get strict guardrails; internal tooling gets lenient ones.

---

## Cost Control & Scaling

### Token Budget System

Each team gets a monthly token budget. The runtime tracks tokens per request (from the LLM provider's response metadata) and enforces:

- **Soft limit**: Log a warning at 80% budget utilization. Team lead gets a Slack notification.
- **Hard limit**: Reject requests at 100%. Return a clear error: "Token budget exceeded for team X this month."

### Cost Reduction Strategies

1. **Semantic caching**: Cache LLM responses for identical (or semantically similar) requests. Cache key = hash of (tool schemas + user request + tool results). Cache hit rate target: 15-30% for common queries like "show pending invoices."

2. **Model routing**: Route simple queries (`list_invoices`) to cheaper models (GPT-4o-mini). Route complex queries (multi-step reasoning, ambiguity resolution) to stronger models (GPT-4o). The router is a lightweight classifier trained on execution traces.

3. **Prompt optimization**: Minimize system prompt tokens. Only include schemas for tools relevant to the request (not all 5 tools for a simple query).

4. **Max iteration cap**: The `max_iterations` config prevents runaway agents from burning tokens in infinite loops. P99 of iterations should be < 5.

### Scaling Characteristics

The runtime is **I/O-bound** (waiting on LLM API calls). Scaling strategy:
- Horizontal scaling of pods (more instances, not bigger instances)
- Connection pooling for LLM API clients
- Async I/O to maximize throughput per pod (currently synchronous — async is the #1 production upgrade)
- Rate limiting per team to prevent noisy neighbors

---

## Developer Experience

### For Teams Building Agents

Product engineers register tools and build agents using a clean SDK:

```python
from light_agent import AgentRuntime, ToolRegistry, RuntimeConfig

# 1. Register your tools
registry = ToolRegistry.from_dict({
    "tools": [
        {
            "name": "get_customer",
            "description": "Look up customer details",
            "parameters": {...},
            "metadata": {"mutating": False}
        }
    ]
})

# 2. Configure runtime
config = RuntimeConfig(
    llm_provider="openai",
    max_iterations=5,
    enable_guardrails=True,
)

# 3. Run
runtime = AgentRuntime(provider, executor, registry, config)
trace = runtime.run("Find customer Acme Corp")
```

### Testing Workflow

1. **Unit tests**: Use `MockProvider` for deterministic testing. No API keys needed.
2. **Integration tests**: Use real provider with recorded fixtures (VCR pattern). Replay API responses for CI.
3. **Evaluation**: Run scenarios against real LLM, compare traces against golden outputs. Automated in CI for model version changes.

### Local Development

```bash
# Quick start
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run all scenarios with mock LLM
python -m light_agent.runner

# Run tests
pytest tests/ -v

# Run with real LLM
export OPENAI_API_KEY="sk-..."
python -c "
from light_agent.runner import run_agent
from light_agent.config import RuntimeConfig
trace = run_agent('Show pending invoices', RuntimeConfig(llm_provider='openai'))
print(trace.summary())
"
```

### Documentation

- **Tool registration guide**: How to define tool schemas with proper metadata
- **Guardrail policies**: What's protected, how to configure per-team
- **Runbook**: Common failure modes and how to diagnose them from execution traces
- **Architecture decision records (ADRs)**: Why we chose ABC over Protocol, why guardrails are in the runtime, etc.
