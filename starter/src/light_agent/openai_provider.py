"""
OpenAI LLM Provider — real GPT integration behind the provider abstraction.

This adapter converts between the runtime's Message types and OpenAI's
API format. It demonstrates that the LLMProvider abstraction isn't
theoretical — swapping from MockProvider to OpenAIProvider is a one-line
config change.

Setup:
    pip install openai
    export OPENAI_API_KEY="sk-..."

    # Then in config:
    config = RuntimeConfig(llm_provider="openai", openai_model="gpt-4o-mini")

The conversion logic lives entirely in this adapter. The agent runtime
never touches OpenAI-specific types.
"""

from __future__ import annotations

import os
from typing import Any

from .llm_provider import LLMProvider, LLMProviderError
from .types import Message, ToolCall


class OpenAIProvider(LLMProvider):
    """
    LLM provider backed by OpenAI's Chat Completions API.

    Handles:
      - Message format conversion (our types ↔ OpenAI format)
      - Tool schema conversion (our format ↔ OpenAI function calling format)
      - Error wrapping (OpenAI errors → LLMProviderError)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise LLMProviderError(
                "OpenAI API key not provided. Set OPENAI_API_KEY env var "
                "or pass api_key parameter.",
                provider="openai",
            )
        try:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key)
        except ImportError:
            raise LLMProviderError(
                "openai package not installed. Run: pip install openai",
                provider="openai",
            )

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        """Send conversation to OpenAI and convert the response back."""
        try:
            openai_messages = self._convert_messages(messages)
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": openai_messages,
            }

            if tools:
                kwargs["tools"] = self._convert_tools(tools)
                kwargs["tool_choice"] = "auto"

            response = self._client.chat.completions.create(**kwargs)
            return self._convert_response(response)

        except Exception as e:
            if "openai" in type(e).__module__:
                raise LLMProviderError(
                    str(e),
                    provider="openai",
                    retryable="rate_limit" in str(e).lower() or "timeout" in str(e).lower(),
                ) from e
            raise

    @property
    def provider_name(self) -> str:
        return f"OpenAI/{self._model}"

    # ------------------------------------------------------------------
    # Format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_messages(messages: list[Message]) -> list[dict]:
        """Convert our Message types to OpenAI's message format."""
        openai_msgs = []
        for msg in messages:
            if msg.role == "user":
                openai_msgs.append({"role": "user", "content": msg.content or ""})
            elif msg.role == "assistant":
                entry: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    entry["content"] = msg.content
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": tc.call_id or f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": __import__("json").dumps(tc.arguments),
                            },
                        }
                        for i, tc in enumerate(msg.tool_calls)
                    ]
                openai_msgs.append(entry)
            elif msg.role == "tool":
                openai_msgs.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "unknown",
                    "content": msg.content or "",
                })
            elif msg.role == "system":
                openai_msgs.append({"role": "system", "content": msg.content or ""})
        return openai_msgs

    @staticmethod
    def _convert_tools(tools: list[dict]) -> list[dict]:
        """Convert our tool schemas to OpenAI's function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {}),
                },
            }
            for tool in tools
        ]

    @staticmethod
    def _convert_response(response: Any) -> Message:
        """Convert OpenAI's response to our Message type."""
        choice = response.choices[0]
        message = choice.message

        tool_calls = None
        if message.tool_calls:
            import json
            tool_calls = [
                ToolCall(
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                    call_id=tc.id,
                )
                for tc in message.tool_calls
            ]

        return Message(
            role="assistant",
            content=message.content,
            tool_calls=tool_calls,
        )
