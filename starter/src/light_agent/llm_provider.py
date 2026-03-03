"""
LLM Provider Abstraction Layer.

This module defines the abstract interface that all LLM backends must implement.
The agent runtime interacts ONLY with this interface — never with provider-specific
SDKs directly. This is the key decoupling point.

Swapping providers is a configuration change, not a code change:
    - MockProvider: deterministic scripted responses for testing
    - OpenAIProvider: real GPT-4o/GPT-4o-mini integration
    - AnthropicProvider: Claude integration (same interface)

Design Decision — Why ABC over Protocol:
    ABC (Abstract Base Class) was chosen because this is infrastructure code
    that other teams will extend. An explicit base class makes the contract
    obvious and surfaces missing method errors at class definition time,
    not at runtime. Protocol would work for structural typing but provides
    weaker guarantees for a platform API.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from .types import Message


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    Every provider must implement a single method: `chat()`.
    This method takes a conversation history (list of Messages) and
    optionally a list of tool schemas, and returns the next assistant
    Message.

    The returned Message will have either:
        - `content` set (the LLM's final answer), or
        - `tool_calls` set (the LLM wants to invoke tools), or
        - both (content is thinking-out-loud; tool_calls take priority)
    """

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        """
        Send a conversation to the LLM and get the next response.

        Args:
            messages: Ordered conversation history (user, assistant, tool messages).
            tools: Optional list of tool schemas in JSON Schema format.
                   Providers should convert these to their native format.

        Returns:
            A Message with role="assistant" containing either content or tool_calls.

        Raises:
            LLMProviderError: If the provider encounters an unrecoverable error.
        """
        ...

    @property
    def provider_name(self) -> str:
        """Human-readable name of this provider (for logging/tracing)."""
        return self.__class__.__name__


class LLMProviderError(Exception):
    """Raised when an LLM provider encounters an error."""

    def __init__(self, message: str, provider: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable
