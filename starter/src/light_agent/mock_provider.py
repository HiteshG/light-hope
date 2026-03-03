"""
Mock LLM Provider — Adapter wrapping the provided MockLLMClient.

This adapter conforms the pre-built MockLLMClient to our LLMProvider
abstraction. We use the Adapter pattern rather than modifying the
original MockLLMClient because:
  1. The challenge says we should NOT need to modify mock_llm.py
  2. It demonstrates that the abstraction works with existing code
  3. It keeps the provided test fixture pristine

Usage:
    provider = MockProvider()
    response = provider.chat(messages, tools)
"""

from __future__ import annotations

from .llm_provider import LLMProvider
from .mock_llm import MockLLMClient
from .types import Message


class MockProvider(LLMProvider):
    """LLM provider backed by scripted responses for deterministic testing."""

    def __init__(self) -> None:
        self._client = MockLLMClient()

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        """Delegate to the scripted mock client."""
        return self._client.chat(messages, tools)

    @property
    def provider_name(self) -> str:
        return "MockLLM"
