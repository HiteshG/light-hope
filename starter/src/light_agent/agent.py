"""
Agent Runtime — the core orchestrator.

This is the engine that sits between the user request and the tools.
It manages the LLM ↔ tool execution loop, enforces guardrails, builds
execution traces, and handles all failure modes.

Architecture:
    User request
         ↓
    AgentRuntime.run()
         ↓
    ┌─────────────────────────────────┐
    │  Loop (max_iterations):         │
    │    1. Send messages to LLM      │
    │    2. If tool_calls:            │
    │       a. Guardrail check        │
    │       b. Execute tool           │
    │       c. Append result          │
    │       d. Continue loop          │
    │    3. If content (no tools):    │
    │       → Final answer, break     │
    └─────────────────────────────────┘
         ↓
    ExecutionTrace (complete record)

Error handling strategy:
    - Tool errors: caught, wrapped in ToolResult(status=ERROR), fed back to LLM
    - LLM errors: caught, set trace.error, return partial trace
    - Max iterations: set trace.error, return what we have
    - Guardrail blocks: feed "blocked" message to LLM, let it explain to user
"""

from __future__ import annotations

import time
import logging
from datetime import datetime, timezone
from typing import Any

from .types import (
    Message,
    ToolCall,
    ToolResult,
    ToolCallStatus,
    StepRecord,
    StepType,
    ExecutionTrace,
)
from .llm_provider import LLMProvider, LLMProviderError
from .tool_registry import ToolRegistry
from .guardrails import GuardrailEngine, GuardrailResult
from .config import RuntimeConfig
from .mock_tools import MockToolExecutor

logger = logging.getLogger(__name__)


class AgentRuntime:
    """
    The core agent runtime — orchestrates LLM decisions and tool execution.

    Design principles:
      - The runtime is provider-agnostic (any LLMProvider implementation works)
      - Every action is recorded in the ExecutionTrace
      - Guardrails are enforced deterministically (not via LLM prompts)
      - Errors are handled gracefully — the runtime never crashes
      - All behavior is configurable via RuntimeConfig
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        tool_executor: MockToolExecutor,
        tool_registry: ToolRegistry,
        config: RuntimeConfig | None = None,
    ) -> None:
        self._llm = llm_provider
        self._executor = tool_executor
        self._registry = tool_registry
        self._config = config or RuntimeConfig()
        self._guardrails = GuardrailEngine(self._registry, self._config)

    def run(self, user_request: str) -> ExecutionTrace:
        """
        Execute the full agent loop for a user request.

        Returns an ExecutionTrace with complete observability data,
        regardless of whether the run succeeded or failed.
        """
        start_time = time.monotonic()
        trace = ExecutionTrace(user_request=user_request)
        step_counter = 0

        # Reset guardrail state for this run
        self._guardrails.reset()

        # Initialize conversation with user message
        messages: list[Message] = [
            Message(role="user", content=user_request)
        ]

        # Get tool schemas for LLM
        tool_schemas = self._registry.get_tool_schemas()

        try:
            for iteration in range(self._config.max_iterations):
                trace.iterations = iteration + 1

                # --- Step: Call LLM ---
                step_counter += 1
                llm_step, llm_response = self._call_llm(
                    messages, tool_schemas, step_counter
                )
                trace.step_records.append(llm_step)
                trace.llm_calls_made += 1

                # Add assistant message to conversation
                messages.append(llm_response)
                trace.steps.append(llm_response)

                # --- Check: Is this the final answer? ---
                if llm_response.content and not llm_response.tool_calls:
                    trace.final_response = llm_response.content
                    break

                # --- Process tool calls ---
                if llm_response.tool_calls:
                    for tool_call in llm_response.tool_calls:
                        step_counter += 1

                        # Guardrail check
                        guardrail_result = self._check_guardrails(
                            tool_call, step_counter, trace
                        )

                        if guardrail_result.action == "block":
                            # Feed block message back to LLM
                            block_msg = Message(
                                role="tool",
                                content=f"BLOCKED: {guardrail_result.reason}",
                                tool_result=ToolResult(
                                    tool_name=tool_call.name,
                                    status=ToolCallStatus.BLOCKED,
                                    error=guardrail_result.reason,
                                ),
                                tool_call_id=tool_call.call_id,
                            )
                            messages.append(block_msg)
                            trace.steps.append(block_msg)
                            continue

                        # Execute tool
                        tool_step, tool_result = self._execute_tool(
                            tool_call, step_counter
                        )
                        trace.step_records.append(tool_step)
                        trace.tool_calls_made += 1

                        # Record mutation for guardrail tracking
                        if tool_result.status == ToolCallStatus.SUCCESS:
                            self._guardrails.record_mutation(tool_call)

                        # Add result to conversation
                        tool_msg = self._tool_result_to_message(
                            tool_call, tool_result
                        )
                        messages.append(tool_msg)
                        trace.steps.append(tool_msg)

                        if guardrail_result.action == "flag":
                            logger.warning(
                                "Guardrail flagged: %s — %s",
                                guardrail_result.policy,
                                guardrail_result.reason,
                            )
            else:
                # Loop exhausted — max iterations reached
                trace.error = (
                    f"Max iterations ({self._config.max_iterations}) exceeded. "
                    f"The agent could not complete the request."
                )
                logger.error("Agent exceeded max iterations for: %s", user_request)

        except LLMProviderError as e:
            trace.error = f"LLM provider error ({e.provider}): {str(e)}"
            logger.error("LLM provider error: %s", e)
        except Exception as e:
            trace.error = f"Runtime error: {str(e)}"
            logger.exception("Unexpected error in agent runtime")

        # Calculate total duration
        trace.total_duration_ms = (time.monotonic() - start_time) * 1000
        return trace

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        messages: list[Message],
        tool_schemas: list[dict],
        step_number: int,
    ) -> tuple[StepRecord, Message]:
        """Call the LLM and record the step."""
        timestamp = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()

        try:
            response = self._llm.chat(messages, tool_schemas)
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            step = StepRecord(
                step_number=step_number,
                step_type=StepType.ERROR,
                timestamp=timestamp,
                duration_ms=duration,
                error_message=f"LLM call failed: {str(e)}",
                model=self._llm.provider_name,
            )
            raise LLMProviderError(
                str(e), provider=self._llm.provider_name
            ) from e

        duration = (time.monotonic() - start) * 1000
        step = StepRecord(
            step_number=step_number,
            step_type=StepType.LLM_CALL,
            timestamp=timestamp,
            duration_ms=duration,
            llm_response=response,
            model=self._llm.provider_name,
        )

        logger.debug(
            "LLM call %d: %s (%.1fms)",
            step_number,
            "tool_calls" if response.tool_calls else "final_answer",
            duration,
        )

        return step, response

    def _execute_tool(
        self,
        tool_call: ToolCall,
        step_number: int,
    ) -> tuple[StepRecord, ToolResult]:
        """Execute a tool call and record the step."""
        timestamp = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()

        try:
            result = self._executor.execute(tool_call.name, tool_call.arguments)
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            result = ToolResult(
                tool_name=tool_call.name,
                status=ToolCallStatus.ERROR,
                error=f"Tool execution exception: {str(e)}",
                duration_ms=duration,
            )

        duration = (time.monotonic() - start) * 1000
        step = StepRecord(
            step_number=step_number,
            step_type=StepType.TOOL_EXECUTION,
            timestamp=timestamp,
            duration_ms=duration,
            tool_call=tool_call,
            tool_result=result,
        )

        logger.debug(
            "Tool %s: %s (%.1fms)",
            tool_call.name,
            result.status.value,
            duration,
        )

        return step, result

    def _check_guardrails(
        self,
        tool_call: ToolCall,
        step_number: int,
        trace: ExecutionTrace,
    ) -> GuardrailResult:
        """Run guardrail checks and record if triggered."""
        timestamp = datetime.now(timezone.utc).isoformat()
        start = time.monotonic()

        result = self._guardrails.check(tool_call)
        duration = (time.monotonic() - start) * 1000

        if result.action != "allow":
            step = StepRecord(
                step_number=step_number,
                step_type=StepType.GUARDRAIL_CHECK,
                timestamp=timestamp,
                duration_ms=duration,
                tool_call=tool_call,
                guardrail_action=result.action,
                guardrail_reason=result.reason,
                guardrail_policy=result.policy,
            )
            trace.step_records.append(step)
            trace.guardrails_triggered.append({
                "policy": result.policy,
                "action": result.action,
                "reason": result.reason,
                "tool": tool_call.name,
            })
            logger.info(
                "Guardrail %s: %s — %s",
                result.action,
                result.policy,
                result.reason,
            )

        return result

    @staticmethod
    def _tool_result_to_message(
        tool_call: ToolCall, result: ToolResult
    ) -> Message:
        """Convert a ToolResult into a Message for the conversation history."""
        if result.status == ToolCallStatus.SUCCESS:
            import json
            content = json.dumps(result.result, default=str)
        else:
            content = f"Error: {result.error}"

        return Message(
            role="tool",
            content=content,
            tool_result=result,
            tool_call_id=tool_call.call_id,
        )
