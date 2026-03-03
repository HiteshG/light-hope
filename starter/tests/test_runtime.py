"""
Unit tests for runtime internals — guardrails, config, provider abstraction.

These tests verify the infrastructure behavior independent of specific
scenarios. They ensure the runtime's safety and reliability mechanisms
work correctly.

Run with:
  pytest tests/ -v
"""

import pytest

from light_agent.types import (
    Message, ToolCall, ToolResult, ToolCallStatus,
    StepRecord, StepType, ExecutionTrace,
)
from light_agent.config import RuntimeConfig
from light_agent.tool_registry import ToolRegistry
from light_agent.guardrails import GuardrailEngine, GuardrailResult
from light_agent.mock_provider import MockProvider
from light_agent.llm_provider import LLMProvider
from light_agent.agent import AgentRuntime
from light_agent.mock_tools import MockToolExecutor
from light_agent.runner import run_agent


# ======================================================================
# Provider Abstraction Tests
# ======================================================================


class TestProviderAbstraction:
    """Verify the LLM provider abstraction works correctly."""

    def test_mock_provider_implements_interface(self):
        provider = MockProvider()
        assert isinstance(provider, LLMProvider)

    def test_mock_provider_returns_message(self):
        provider = MockProvider()
        messages = [Message(role="user", content="Show me all unpaid invoices over €5,000")]
        response = provider.chat(messages)
        assert isinstance(response, Message)
        assert response.role == "assistant"

    def test_mock_provider_has_name(self):
        provider = MockProvider()
        assert provider.provider_name == "MockLLM"

    def test_provider_abstraction_is_enforced(self):
        """Cannot instantiate LLMProvider directly."""
        with pytest.raises(TypeError):
            LLMProvider()


# ======================================================================
# Tool Registry Tests
# ======================================================================


class TestToolRegistry:
    """Verify tool metadata lookups."""

    def test_loads_all_tools(self):
        registry = ToolRegistry()
        assert len(registry.tool_names) == 5
        assert "list_invoices" in registry.tool_names
        assert "approve_invoice" in registry.tool_names

    def test_mutating_classification(self):
        registry = ToolRegistry()
        assert registry.is_mutating("approve_invoice") is True
        assert registry.is_mutating("send_notification") is True
        assert registry.is_mutating("list_invoices") is False
        assert registry.is_mutating("get_invoice") is False
        assert registry.is_mutating("get_current_user") is False

    def test_irreversible_classification(self):
        registry = ToolRegistry()
        assert registry.is_irreversible("approve_invoice") is True
        assert registry.is_irreversible("list_invoices") is False

    def test_risk_classification(self):
        registry = ToolRegistry()
        assert registry.classify_risk("list_invoices") == "safe"
        assert registry.classify_risk("approve_invoice") == "dangerous"
        assert registry.classify_risk("send_notification") == "cautious"
        assert registry.classify_risk("nonexistent_tool") == "unknown"

    def test_tool_schemas_format(self):
        registry = ToolRegistry()
        schemas = registry.get_tool_schemas()
        assert len(schemas) == 5
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema

    def test_unknown_tool_detection(self):
        registry = ToolRegistry()
        assert registry.is_known("list_invoices") is True
        assert registry.is_known("hack_the_planet") is False


# ======================================================================
# Guardrail Engine Tests
# ======================================================================


class TestGuardrailEngine:
    """Verify guardrail policies work correctly."""

    def setup_method(self):
        self.registry = ToolRegistry()
        self.config = RuntimeConfig(enable_guardrails=True, max_mutations_without_confirmation=1)
        self.engine = GuardrailEngine(self.registry, self.config)

    def test_allows_read_only_tools(self):
        call = ToolCall(name="list_invoices", arguments={"status": "pending"})
        result = self.engine.check(call)
        assert result.action == "allow"

    def test_flags_irreversible_mutations(self):
        call = ToolCall(name="approve_invoice", arguments={"invoice_id": "INV-001"})
        result = self.engine.check(call)
        # First mutation is flagged but allowed
        assert result.action in ("allow", "flag")

    def test_blocks_bulk_mutations(self):
        # First mutation — allowed
        call1 = ToolCall(name="approve_invoice", arguments={"invoice_id": "INV-001"})
        result1 = self.engine.check(call1)
        assert result1.action != "block"

        # Record that the first mutation happened
        self.engine.record_mutation(call1)

        # Second mutation — blocked (bulk mutation guard)
        call2 = ToolCall(name="approve_invoice", arguments={"invoice_id": "INV-002"})
        result2 = self.engine.check(call2)
        assert result2.action == "block"
        assert result2.policy == "bulk_mutation_guard"

    def test_blocks_unknown_tools(self):
        call = ToolCall(name="hack_the_planet", arguments={})
        result = self.engine.check(call)
        assert result.action == "block"
        assert result.policy == "unknown_tool"

    def test_reset_clears_mutation_count(self):
        call = ToolCall(name="approve_invoice", arguments={"invoice_id": "INV-001"})
        self.engine.record_mutation(call)
        assert self.engine._mutation_count == 1
        self.engine.reset()
        assert self.engine._mutation_count == 0

    def test_guardrails_disabled(self):
        config = RuntimeConfig(enable_guardrails=False)
        engine = GuardrailEngine(self.registry, config)
        call = ToolCall(name="hack_the_planet", arguments={})
        result = engine.check(call)
        assert result.action == "allow"


# ======================================================================
# Configuration Tests
# ======================================================================


class TestConfig:
    """Verify configuration defaults are sensible."""

    def test_defaults(self):
        config = RuntimeConfig()
        assert config.max_iterations == 10
        assert config.tool_timeout_seconds == 30.0
        assert config.max_retries == 2
        assert config.enable_guardrails is True
        assert config.max_mutations_without_confirmation == 1
        assert config.llm_provider == "mock"

    def test_custom_config(self):
        config = RuntimeConfig(
            max_iterations=5,
            enable_guardrails=False,
            llm_provider="openai",
        )
        assert config.max_iterations == 5
        assert config.enable_guardrails is False
        assert config.llm_provider == "openai"


# ======================================================================
# Execution Trace Tests
# ======================================================================


class TestExecutionTrace:
    """Verify trace serialization and summary."""

    def test_trace_summary(self):
        trace = ExecutionTrace(
            user_request="test",
            final_response="done",
            tool_calls_made=2,
            llm_calls_made=3,
            total_duration_ms=150.5,
        )
        summary = trace.summary()
        assert "test" in summary
        assert "OK" in summary
        assert "150ms" in summary

    def test_trace_to_dict(self):
        trace = ExecutionTrace(user_request="test", final_response="done")
        d = trace.to_dict()
        assert d["user_request"] == "test"
        assert d["final_response"] == "done"
        assert "steps" in d

    def test_error_trace_summary(self):
        trace = ExecutionTrace(
            user_request="test",
            error="Max iterations exceeded",
            total_duration_ms=100.0,
        )
        summary = trace.summary()
        assert "ERROR" in summary
        assert "Max iterations" in summary


# ======================================================================
# Agent Runtime Edge Cases
# ======================================================================


class TestAgentRuntimeEdgeCases:
    """Verify runtime handles edge cases gracefully."""

    def test_max_iterations_protection(self):
        """Runtime should stop after max_iterations even if LLM keeps requesting tools."""
        config = RuntimeConfig(max_iterations=1)
        result = run_agent(
            "Find the largest pending invoice from Acme Corp and approve it",
            config=config,
        )
        # With max_iterations=1, the agent can only do 1 loop
        # It should stop and report the limitation
        assert result.iterations <= 1

    def test_runtime_produces_trace_even_on_error(self):
        """Even failed runs should have some trace data."""
        config = RuntimeConfig(max_iterations=1)
        result = run_agent("Approve invoice INV-9999", config=config)
        assert result.total_duration_ms is not None
        assert result.total_duration_ms > 0
        assert len(result.step_records) >= 1

    def test_tool_error_is_in_trace(self):
        """Tool errors should appear in the execution trace."""
        result = run_agent("Approve invoice INV-9999")
        tool_steps = [
            s for s in result.step_records
            if s.step_type == StepType.TOOL_EXECUTION
        ]
        error_steps = [
            s for s in tool_steps
            if s.tool_result and s.tool_result.status == ToolCallStatus.ERROR
        ]
        assert len(error_steps) >= 1
