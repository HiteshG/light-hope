"""
Light Agent Runtime — Entry Point & Factory.

This module wires all components together and provides the main entry
point for running the agent. It acts as a factory — constructing the
right LLMProvider, ToolExecutor, ToolRegistry, and AgentRuntime based
on configuration.

Usage:
    from light_agent.runner import run_agent
    trace = run_agent("Show me all unpaid invoices over €5,000")
    print(trace.summary())

    # Or with custom config:
    from light_agent.config import RuntimeConfig
    config = RuntimeConfig(llm_provider="openai", max_iterations=5)
    trace = run_agent("...", config=config)
"""

import json
import logging

from light_agent.config import RuntimeConfig
from light_agent.llm_provider import LLMProvider
from light_agent.mock_provider import MockProvider
from light_agent.mock_tools import MockToolExecutor
from light_agent.tool_registry import ToolRegistry
from light_agent.agent import AgentRuntime
from light_agent.types import ExecutionTrace

logger = logging.getLogger(__name__)


def _create_provider(config: RuntimeConfig) -> LLMProvider:
    """
    Factory for LLM providers based on configuration.

    Swapping providers is a config change, not a code change:
        RuntimeConfig(llm_provider="mock")   → MockProvider (deterministic tests)
        RuntimeConfig(llm_provider="openai") → OpenAIProvider (real GPT)
    """
    if config.llm_provider == "mock":
        return MockProvider()

    if config.llm_provider == "openai":
        from light_agent.openai_provider import OpenAIProvider
        return OpenAIProvider(
            model=config.openai_model,
            api_key=config.openai_api_key,
        )

    raise ValueError(
        f"Unknown LLM provider: '{config.llm_provider}'. "
        f"Available: mock, openai"
    )


def run_agent(
    user_request: str,
    config: RuntimeConfig | None = None,
) -> ExecutionTrace:
    """
    Run the agent for a given user request.

    This is the main public API. It:
      1. Creates/configures all components
      2. Runs the agent loop
      3. Returns a structured ExecutionTrace

    Args:
        user_request: Natural-language request from the user.
        config: Optional runtime configuration. Uses defaults if not provided.

    Returns:
        ExecutionTrace with complete observability data.
    """
    config = config or RuntimeConfig()
    provider = _create_provider(config)
    executor = MockToolExecutor()
    registry = ToolRegistry()
    runtime = AgentRuntime(provider, executor, registry, config)
    return runtime.run(user_request)


# Quick manual test
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scenarios = [
        ("Scenario 1 — Simple Query",
         "Show me all unpaid invoices over €5,000"),
        ("Scenario 2 — Multi-Step",
         "Find the largest pending invoice from Acme Corp and approve it"),
        ("Scenario 3 — Error Handling",
         "Approve invoice INV-9999"),
        ("Scenario 4 — Bulk Mutation",
         "Approve all pending invoices"),
        ("Scenario 5 — Ambiguity",
         "What's the status of the Globex invoice?"),
    ]

    for name, request in scenarios:
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"  REQUEST: {request}")
        print(f"{'='*70}")

        try:
            trace = run_agent(request)
            print(f"\n{trace.summary()}")

            # Print step-by-step trace
            print(f"\n--- Execution Steps ---")
            for step in trace.step_records:
                d = step.to_dict()
                step_type = d["type"]
                dur = d["duration_ms"]

                if step_type == "llm_call":
                    if step.llm_response and step.llm_response.tool_calls:
                        tools = [tc.name for tc in step.llm_response.tool_calls]
                        print(f"  [{step.step_number}] LLM → tool_calls: {tools} ({dur}ms)")
                    else:
                        print(f"  [{step.step_number}] LLM → final answer ({dur}ms)")
                elif step_type == "tool_execution":
                    tc = d.get("tool_call", {})
                    tr = d.get("tool_result", {})
                    print(f"  [{step.step_number}] TOOL {tc.get('name')}: {tr.get('status')} ({dur}ms)")
                elif step_type == "guardrail_check":
                    gr = d.get("guardrail", {})
                    print(f"  [{step.step_number}] GUARDRAIL {gr.get('policy')}: {gr.get('action')} — {gr.get('reason')}")

            if trace.guardrails_triggered:
                print(f"\n--- Guardrails Triggered ---")
                for g in trace.guardrails_triggered:
                    print(f"  ⚠ {g['policy']}: {g['action']} — {g['reason']}")

            print(f"\n--- Final Response ---")
            print(f"  {trace.final_response}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
