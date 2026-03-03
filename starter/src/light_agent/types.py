"""
Shared types for the Light Agent Runtime.

These types form the contract between all components:
- Message: Conversation threading between LLM and tools
- ToolCall / ToolResult: Tool execution request/response
- StepRecord: Rich observability for each step in the agent loop
- ExecutionTrace: Complete record of a single agent run
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Tool execution types
# ---------------------------------------------------------------------------

class ToolCallStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"  # Guardrail prevented execution


@dataclass
class ToolCall:
    """A request from the LLM to execute a specific tool."""

    name: str
    arguments: dict[str, Any]
    call_id: str | None = None  # Optional ID to correlate calls with results


@dataclass
class ToolResult:
    """The outcome of executing a tool."""

    tool_name: str
    status: ToolCallStatus
    result: Any | None = None
    error: str | None = None
    duration_ms: float | None = None


# ---------------------------------------------------------------------------
# Conversation / message types
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """
    A single message in the agent's conversation history.

    role:
        - "user"       — the original user request
        - "assistant"   — an LLM response (may contain tool_calls and/or content)
        - "tool"        — the result of a tool execution
        - "system"      — system-level instructions (e.g., guardrail messages)
    """

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None
    tool_call_id: str | None = None  # Links tool result back to the requesting call


# ---------------------------------------------------------------------------
# Step records — rich observability
# ---------------------------------------------------------------------------

class StepType(Enum):
    """What kind of step occurred in the agent loop."""
    LLM_CALL = "llm_call"
    TOOL_EXECUTION = "tool_execution"
    GUARDRAIL_CHECK = "guardrail_check"
    ERROR = "error"


@dataclass
class StepRecord:
    """
    A single atomic step in the agent execution.

    More granular than Message — each step has a type, timing, and
    metadata specific to what happened. This is the primary
    observability output.
    """

    step_number: int
    step_type: StepType
    timestamp: str  # ISO-8601
    duration_ms: float = 0.0

    # LLM call details
    llm_response: Message | None = None
    model: str | None = None

    # Tool execution details
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None

    # Guardrail details
    guardrail_action: str | None = None  # "allow", "block", "flag"
    guardrail_reason: str | None = None
    guardrail_policy: str | None = None  # Which guardrail triggered

    # Error details
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Serialize for logging / JSON output."""
        d: dict[str, Any] = {
            "step": self.step_number,
            "type": self.step_type.value,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
        }
        if self.tool_call:
            d["tool_call"] = {
                "name": self.tool_call.name,
                "arguments": self.tool_call.arguments,
                "call_id": self.tool_call.call_id,
            }
        if self.tool_result:
            d["tool_result"] = {
                "tool_name": self.tool_result.tool_name,
                "status": self.tool_result.status.value,
                "duration_ms": self.tool_result.duration_ms,
            }
            if self.tool_result.error:
                d["tool_result"]["error"] = self.tool_result.error
        if self.guardrail_action:
            d["guardrail"] = {
                "action": self.guardrail_action,
                "reason": self.guardrail_reason,
                "policy": self.guardrail_policy,
            }
        if self.error_message:
            d["error"] = self.error_message
        if self.model:
            d["model"] = self.model
        return d


# ---------------------------------------------------------------------------
# Execution trace — complete observability record
# ---------------------------------------------------------------------------

@dataclass
class ExecutionTrace:
    """
    Structured record of a single agent run.

    This is the observability output. A good trace lets you understand
    exactly what happened — which tools were called, in what order, what
    the LLM decided at each step, and how long it all took.
    """

    user_request: str = ""
    # Backward-compatible message list (used by test stubs)
    steps: list[Message] = field(default_factory=list)
    # Rich step records for deep observability
    step_records: list[StepRecord] = field(default_factory=list)
    final_response: str | None = None
    tool_calls_made: int = 0
    llm_calls_made: int = 0
    iterations: int = 0
    total_duration_ms: float | None = None
    error: str | None = None
    guardrails_triggered: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable summary of the trace."""
        lines = [
            f"Request: {self.user_request}",
            f"Status: {'ERROR — ' + (self.error or 'unknown') if self.error else 'OK'}",
            f"Iterations: {self.iterations}",
            f"LLM calls: {self.llm_calls_made}",
            f"Tool calls: {self.tool_calls_made}",
            f"Duration: {self.total_duration_ms:.0f}ms" if self.total_duration_ms else "Duration: N/A",
        ]
        if self.guardrails_triggered:
            lines.append(f"Guardrails triggered: {len(self.guardrails_triggered)}")
        lines.append(f"Response: {self.final_response or '(none)'}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Full serializable representation."""
        return {
            "user_request": self.user_request,
            "final_response": self.final_response,
            "tool_calls_made": self.tool_calls_made,
            "llm_calls_made": self.llm_calls_made,
            "iterations": self.iterations,
            "total_duration_ms": round(self.total_duration_ms, 2) if self.total_duration_ms else None,
            "error": self.error,
            "guardrails_triggered": self.guardrails_triggered,
            "steps": [s.to_dict() for s in self.step_records],
        }
