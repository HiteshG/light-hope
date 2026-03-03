"""
Runtime Configuration — all tunable parameters in one place.

Every behavioral parameter is configurable here. Nothing is hardcoded
in the agent loop. This makes the runtime:
  - Testable (override config in tests)
  - Tunable (adjust per-team or per-use-case)
  - Transparent (all knobs are documented)

In production, this would be loaded from environment variables, a config
file (YAML/TOML), or a service like LaunchDarkly. The dataclass is the
SCHEMA — the loading mechanism is separate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeConfig:
    """
    Configuration for the Light Agent Runtime.

    All values have sensible defaults that work for the test scenarios.
    Override for production or specific use cases.
    """

    # --- LLM Provider ---
    llm_provider: str = "mock"
    """Which LLM provider to use: "mock", "openai", "anthropic"."""

    openai_model: str = "gpt-4o-mini"
    """Model name for OpenAI provider."""

    openai_api_key: str | None = None
    """API key for OpenAI (default: reads from OPENAI_API_KEY env var)."""

    # --- Agent Loop Limits ---
    max_iterations: int = 10
    """Maximum LLM ↔ tool loop iterations before forced termination.

    Why 10? The test scenarios need at most 2 iterations. 10 gives headroom
    for real-world multi-step tasks while preventing infinite loops.
    A run-away agent burning $50 in API calls is much worse than stopping early.
    """

    # --- Tool Execution ---
    tool_timeout_seconds: float = 30.0
    """Maximum time to wait for a single tool execution.

    If a tool hangs (e.g., downstream API is down), we don't want the
    entire agent to block. 30s is generous — most tools should complete
    in under 1s.
    """

    max_retries: int = 2
    """Maximum retries for failed tool calls.

    Only retries on transient errors (timeouts, 5xx). Permanent errors
    (not found, invalid state) are NOT retried — they're fed back to
    the LLM immediately.
    """

    retry_delay_seconds: float = 1.0
    """Delay between retries (simple constant backoff).

    In production, use exponential backoff with jitter:
      delay = min(base * 2^attempt + random(0, jitter), max_delay)
    """

    # --- Guardrails ---
    enable_guardrails: bool = True
    """Master switch for the guardrail engine.

    Set to False for testing or trusted internal tools.
    When disabled, all tool calls execute without checks.
    """

    max_mutations_without_confirmation: int = 1
    """How many mutating tools can execute before requiring confirmation.

    Why 1? "One mutation is deliberate; multiple mutations in a single
    request might be a mistake." Scenario 2 approves one invoice — fine.
    Scenario 4 tries to approve five — blocked after the first.

    Set higher for teams that need batch operations (e.g., data migrations).
    """

    # --- Observability ---
    log_level: str = "INFO"
    """Logging level for the runtime."""

    include_tool_results_in_trace: bool = True
    """Whether to include full tool results in the execution trace.

    Disable for sensitive data (PII, financial details) in production logs.
    """
