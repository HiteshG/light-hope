"""
Tool Registry — metadata-aware tool schema management.

Separates tool KNOWLEDGE (schemas, metadata, safety classification) from
tool EXECUTION (MockToolExecutor). This separation is critical:

  - The guardrail engine needs to know "is this tool mutating?" BEFORE execution
  - The LLM needs tool schemas to decide which tools to call
  - The executor just runs the call and returns a result

If these were combined, the guardrail engine would depend on the executor,
creating a circular responsibility. The registry is the single source of
truth for "what tools exist and what are their properties."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_DEFAULT_TOOLS_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "tools.json"


class ToolRegistry:
    """
    Loads tool definitions from tools.json and provides metadata lookups.

    Responsibilities:
      - Parse and store tool schemas
      - Provide tool schemas in LLM-consumable format
      - Answer metadata queries (is_mutating, is_irreversible, requires_confirmation)

    NOT responsible for:
      - Executing tools (that's ToolExecutor)
      - Enforcing policies (that's GuardrailEngine)
    """

    def __init__(self, tools_path: str | Path | None = None) -> None:
        path = Path(tools_path) if tools_path else _DEFAULT_TOOLS_PATH
        with open(path) as f:
            data = json.load(f)
        self._tools: dict[str, dict] = {}
        for tool in data.get("tools", []):
            self._tools[tool["name"]] = tool

    @property
    def tool_names(self) -> list[str]:
        """All registered tool names."""
        return list(self._tools.keys())

    def get_tool_schemas(self) -> list[dict]:
        """
        Return tool schemas formatted for LLM consumption.

        Returns the full schema including name, description, and parameters
        for each tool. The LLM uses these to decide which tool to call.
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            }
            for tool in self._tools.values()
        ]

    def get_tool(self, tool_name: str) -> dict | None:
        """Get full tool definition by name, or None if not registered."""
        return self._tools.get(tool_name)

    def get_metadata(self, tool_name: str) -> dict:
        """Get tool metadata (mutating, irreversible, etc.)."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return {}
        return tool.get("metadata", {})

    def is_mutating(self, tool_name: str) -> bool:
        """Check if a tool modifies state (writes, updates, deletes)."""
        return self.get_metadata(tool_name).get("mutating", False)

    def is_irreversible(self, tool_name: str) -> bool:
        """Check if a tool's action cannot be undone."""
        return self.get_metadata(tool_name).get("irreversible", False)

    def requires_confirmation(self, tool_name: str) -> bool:
        """Check if a tool needs explicit user confirmation before execution."""
        return self.get_metadata(tool_name).get("requires_confirmation", False)

    def is_idempotent(self, tool_name: str) -> bool:
        """Check if a tool can be safely retried (same call = same result)."""
        return self.get_metadata(tool_name).get("idempotent", True)

    def is_known(self, tool_name: str) -> bool:
        """Check if a tool exists in the registry."""
        return tool_name in self._tools

    def classify_risk(self, tool_name: str) -> str:
        """
        Classify a tool's risk level for observability.

        Returns:
            "safe"       — read-only, no side effects
            "cautious"   — mutating but reversible/idempotent
            "dangerous"  — mutating + irreversible
            "unknown"    — tool not in registry
        """
        if not self.is_known(tool_name):
            return "unknown"
        if not self.is_mutating(tool_name):
            return "safe"
        if self.is_irreversible(tool_name):
            return "dangerous"
        return "cautious"
