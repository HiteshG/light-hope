"""
Guardrail Engine — pre-execution safety checks.

Guardrails sit BETWEEN the LLM's decision and the tool execution. This is
the critical trust boundary in the agent runtime. Even if the LLM decides
to "approve all 5 invoices," the guardrail engine intercepts and blocks
the bulk mutation.

Why guardrails MUST be in the runtime (not the prompt):
  - LLM prompts are advisory — they can be ignored, jailbroken, or hallucinated past
  - The runtime is deterministic — if the policy says "block," it blocks. Period.
  - Defense-in-depth: the prompt is layer 1, the runtime is layer 2

Guardrail policies:
  1. MutationGuard — flags/blocks mutating operations on irreversible tools
  2. BulkMutationGuard — detects and blocks batch mutations in a single run
  3. UnknownToolGuard — blocks calls to tools not in the registry
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .types import ToolCall
from .tool_registry import ToolRegistry
from .config import RuntimeConfig


@dataclass
class GuardrailResult:
    """
    Outcome of a guardrail check.

    action:
        "allow"  — proceed with execution
        "block"  — do NOT execute; return this result to the LLM
        "flag"   — execute but log a warning (for observability)
    """
    action: str  # "allow" | "block" | "flag"
    policy: str = ""  # Which guardrail policy made this decision
    reason: str | None = None
    details: dict = field(default_factory=dict)


class GuardrailEngine:
    """
    Orchestrates pre-execution safety checks.

    Each check method returns a GuardrailResult. The engine runs all
    applicable checks and returns the most restrictive result
    (block > flag > allow).
    """

    def __init__(self, registry: ToolRegistry, config: RuntimeConfig) -> None:
        self._registry = registry
        self._config = config
        # Track mutations within a single run for bulk detection
        self._mutation_count: int = 0
        self._mutated_tools: list[str] = []

    def reset(self) -> None:
        """Reset per-run state. Call at the start of each agent run."""
        self._mutation_count = 0
        self._mutated_tools = []

    def check(self, tool_call: ToolCall, context: dict | None = None) -> GuardrailResult:
        """
        Run all applicable guardrails for a tool call.

        Returns the most restrictive result across all policies.

        Args:
            tool_call: The tool call to check.
            context: Optional context (e.g., conversation history, user info).
        """
        if not self._config.enable_guardrails:
            return GuardrailResult(action="allow", policy="guardrails_disabled")

        context = context or {}
        checks = [
            self._check_unknown_tool,
            self._check_mutation_guard,
            self._check_bulk_mutation_guard,
        ]

        most_restrictive = GuardrailResult(action="allow", policy="default")
        for check_fn in checks:
            result = check_fn(tool_call, context)
            if result is not None:
                if result.action == "block":
                    return result  # Immediate block — no need to continue
                if result.action == "flag" and most_restrictive.action == "allow":
                    most_restrictive = result

        return most_restrictive

    def record_mutation(self, tool_call: ToolCall) -> None:
        """Record that a mutating tool was executed (call after execution)."""
        if self._registry.is_mutating(tool_call.name):
            self._mutation_count += 1
            self._mutated_tools.append(tool_call.name)

    # ------------------------------------------------------------------
    # Individual guardrail policies
    # ------------------------------------------------------------------

    def _check_unknown_tool(
        self, tool_call: ToolCall, context: dict
    ) -> GuardrailResult | None:
        """Block calls to tools not in the registry."""
        if not self._registry.is_known(tool_call.name):
            return GuardrailResult(
                action="block",
                policy="unknown_tool",
                reason=f"Tool '{tool_call.name}' is not registered. "
                       f"Available tools: {', '.join(self._registry.tool_names)}",
            )
        return None

    def _check_mutation_guard(
        self, tool_call: ToolCall, context: dict
    ) -> GuardrailResult | None:
        """Flag irreversible mutating operations for observability."""
        if not self._registry.is_mutating(tool_call.name):
            return None  # Read-only — no guardrail needed

        risk = self._registry.classify_risk(tool_call.name)
        if risk == "dangerous":
            return GuardrailResult(
                action="flag",
                policy="mutation_guard",
                reason=f"Tool '{tool_call.name}' is mutating and irreversible.",
                details={
                    "risk_level": risk,
                    "tool": tool_call.name,
                    "arguments": tool_call.arguments,
                },
            )
        return None

    def _check_bulk_mutation_guard(
        self, tool_call: ToolCall, context: dict
    ) -> GuardrailResult | None:
        """
        Block bulk mutations — when multiple mutating calls happen in one run.

        Logic: If we've already executed N mutating calls and another is requested,
        block it. The threshold is configurable via RuntimeConfig.

        Why this catches Scenario 4: The LLM tries to approve 5 invoices.
        After the first approve, the guard blocks the rest and tells the LLM
        to ask for confirmation.
        """
        if not self._registry.is_mutating(tool_call.name):
            return None

        threshold = self._config.max_mutations_without_confirmation
        if self._mutation_count >= threshold:
            return GuardrailResult(
                action="block",
                policy="bulk_mutation_guard",
                reason=(
                    f"Bulk mutation blocked: {self._mutation_count} mutating "
                    f"operation(s) already executed in this run "
                    f"(threshold: {threshold}). Additional mutations require "
                    f"explicit user confirmation."
                ),
                details={
                    "mutation_count": self._mutation_count,
                    "threshold": threshold,
                    "blocked_tool": tool_call.name,
                    "previous_mutations": self._mutated_tools.copy(),
                },
            )
        return None
