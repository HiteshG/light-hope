"""
Mock tool executor — simulates the Light finance API.

This reads the mock data from data/mock_data.json and executes tool calls
against it. The implementations intentionally mirror what a real API would
return, including realistic error messages.

You should NOT need to modify this file (but you can if you want to).
Your runtime should treat this as an opaque executor: give it a tool name
and arguments, get back a result or error.
"""

from __future__ import annotations

import json
import time
import random
from pathlib import Path
from typing import Any

from .types import ToolResult, ToolCallStatus

# Default path to mock data — works when running from the starter/ directory
_DEFAULT_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "mock_data.json"


class MockToolExecutor:
    """
    Executes tool calls against mock data.

    Usage:
        executor = MockToolExecutor()
        result = executor.execute("list_invoices", {"status": "pending"})
    """

    def __init__(self, data_path: str | Path | None = None):
        path = Path(data_path) if data_path else _DEFAULT_DATA_PATH
        with open(path) as f:
            self._data = json.load(f)
        self._invoices: list[dict] = self._data["invoices"]
        self._current_user: dict = self._data["current_user"]

    def execute(self, tool_name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Execute a tool by name. Returns a ToolResult with status and data."""
        arguments = arguments or {}
        handler = getattr(self, f"_tool_{tool_name}", None)

        if handler is None:
            return ToolResult(
                tool_name=tool_name,
                status=ToolCallStatus.ERROR,
                error=f"Unknown tool: '{tool_name}'. Available tools: "
                      f"{', '.join(self.available_tools())}",
            )

        start = time.monotonic()
        try:
            # Simulate a small amount of latency (5–50 ms)
            time.sleep(random.uniform(0.005, 0.05))
            result = handler(**arguments)
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=tool_name,
                status=ToolCallStatus.SUCCESS,
                result=result,
                duration_ms=round(elapsed, 2),
            )
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return ToolResult(
                tool_name=tool_name,
                status=ToolCallStatus.ERROR,
                error=str(e),
                duration_ms=round(elapsed, 2),
            )

    def available_tools(self) -> list[str]:
        """Return the names of all available tools."""
        return [
            name.removeprefix("_tool_")
            for name in dir(self)
            if name.startswith("_tool_")
        ]

    # ------------------------------------------------------------------
    # Tool implementations
    # ------------------------------------------------------------------

    def _tool_list_invoices(
        self,
        status: str | None = None,
        vendor: str | None = None,
        min_amount: float | None = None,
        max_amount: float | None = None,
        currency: str | None = None,
    ) -> list[dict]:
        """Search and filter invoices."""
        results = list(self._invoices)  # shallow copy

        if status and status != "all":
            if status == "unpaid":
                # Convenience alias: everything that isn't "paid"
                results = [inv for inv in results if inv["status"] != "paid"]
            else:
                results = [inv for inv in results if inv["status"] == status]

        if vendor:
            vendor_lower = vendor.lower()
            results = [
                inv for inv in results if vendor_lower in inv["vendor"].lower()
            ]

        if min_amount is not None:
            results = [inv for inv in results if inv["amount"] >= min_amount]

        if max_amount is not None:
            results = [inv for inv in results if inv["amount"] <= max_amount]

        if currency:
            results = [
                inv for inv in results if inv["currency"].upper() == currency.upper()
            ]

        # Return summaries (strip line_items for the list view)
        return [
            {k: v for k, v in inv.items() if k != "line_items"} for inv in results
        ]

    def _tool_get_invoice(self, invoice_id: str) -> dict:
        """Get full invoice details including line items."""
        for inv in self._invoices:
            if inv["id"] == invoice_id:
                return inv
        raise ValueError(f"Invoice not found: {invoice_id}")

    def _tool_approve_invoice(self, invoice_id: str) -> dict:
        """Approve a pending invoice. Raises on invalid state transitions."""
        for inv in self._invoices:
            if inv["id"] == invoice_id:
                if inv["status"] != "pending":
                    raise ValueError(
                        f"Cannot approve invoice {invoice_id}: current status is "
                        f"'{inv['status']}'. Only 'pending' invoices can be approved."
                    )
                if inv["amount"] > self._current_user.get("approval_limit", float("inf")):
                    raise ValueError(
                        f"Cannot approve invoice {invoice_id}: amount €{inv['amount']:,.2f} "
                        f"exceeds your approval limit of "
                        f"€{self._current_user['approval_limit']:,.2f}."
                    )
                return {
                    "success": True,
                    "invoice_id": invoice_id,
                    "previous_status": "pending",
                    "new_status": "approved",
                    "approved_by": self._current_user["name"],
                    "message": f"Invoice {invoice_id} has been approved.",
                }
        raise ValueError(f"Invoice not found: {invoice_id}")

    def _tool_send_notification(self, channel: str, message: str) -> dict:
        """Send a Slack notification."""
        if not channel:
            raise ValueError("Parameter 'channel' is required and cannot be empty.")
        if not message:
            raise ValueError("Parameter 'message' is required and cannot be empty.")
        return {
            "success": True,
            "channel": channel,
            "message_preview": message[:100],
            "timestamp": "2026-02-27T10:30:00Z",
        }

    def _tool_get_current_user(self) -> dict:
        """Get the current user's profile and permissions."""
        return self._current_user
