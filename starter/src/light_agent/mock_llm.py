"""
Mock LLM client — scripted responses for testing your agent runtime.

This provides deterministic, pre-scripted responses for each test scenario
so you can develop and test your runtime without needing an API key.

IMPORTANT:
  - This is a convenience for testing, NOT a constraint on your design.
  - Your runtime should interact with the LLM through an abstraction
    (interface / protocol / base class) that this mock implements.
  - You're encouraged to also wire up a real LLM provider (OpenAI,
    Anthropic, etc.) behind the same abstraction.
  - The mock routes based on keywords in the user message. It won't
    handle arbitrary requests — that's what a real LLM is for.
"""

from __future__ import annotations

from .types import Message, ToolCall


class MockLLMClient:
    """
    A scripted LLM that returns canned responses for the five test scenarios.

    It inspects the conversation history to figure out which scenario is
    running and how far along it is (based on how many tool results have
    come back), then returns the next scripted response.
    """

    def chat(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
    ) -> Message:
        """
        Given a conversation history, return the next assistant message.

        Returns either:
          - A message with tool_calls (the LLM wants to use a tool), or
          - A message with content (the LLM's final answer to the user).
        """
        user_msg = self._last_user_message(messages)
        tool_result_count = self._count_tool_results(messages)

        # --- Route to the matching scenario ---

        if self._matches_scenario_1(user_msg):
            return self._scenario_1(tool_result_count)

        if self._matches_scenario_2(user_msg):
            return self._scenario_2(tool_result_count)

        if self._matches_scenario_3(user_msg):
            return self._scenario_3(tool_result_count)

        if self._matches_scenario_4(user_msg):
            return self._scenario_4(tool_result_count)

        if self._matches_scenario_5(user_msg):
            return self._scenario_5(tool_result_count)

        # Fallback — no matching scenario
        return Message(
            role="assistant",
            content="I'm not sure how to help with that. Could you rephrase your request?",
        )

    # ------------------------------------------------------------------
    # Scenario matchers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_scenario_1(msg: str) -> bool:
        lower = msg.lower()
        return "unpaid" in lower and ("5,000" in msg or "5000" in msg)

    @staticmethod
    def _matches_scenario_2(msg: str) -> bool:
        lower = msg.lower()
        return "largest" in lower and "acme" in lower and "approve" in lower

    @staticmethod
    def _matches_scenario_3(msg: str) -> bool:
        return "INV-9999" in msg

    @staticmethod
    def _matches_scenario_4(msg: str) -> bool:
        lower = msg.lower()
        return "approve all" in lower and "pending" in lower

    @staticmethod
    def _matches_scenario_5(msg: str) -> bool:
        lower = msg.lower()
        return "globex" in lower and "status" in lower

    # ------------------------------------------------------------------
    # Scripted scenario responses
    # ------------------------------------------------------------------

    def _scenario_1(self, tool_results_so_far: int) -> Message:
        """Simple query — list unpaid invoices over €5,000."""
        if tool_results_so_far == 0:
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="list_invoices",
                        arguments={"status": "unpaid", "min_amount": 5000},
                        call_id="call_1",
                    )
                ],
            )
        return Message(
            role="assistant",
            content=(
                "Here are your unpaid invoices over €5,000:\n\n"
                "| Invoice | Vendor | Amount | Status | Due Date |\n"
                "|---------|--------|--------|--------|----------|\n"
                "| INV-001 | Acme Corp | €15,000.00 | Pending | 2026-03-15 |\n"
                "| INV-003 | Globex Industries | €8,750.00 | Pending | 2026-03-01 |\n"
                "| INV-005 | Initech Solutions | €42,000.00 | Pending | 2026-04-01 |\n"
                "| INV-004 | Globex Industries | €8,750.00 | Overdue | 2026-01-15 |\n"
                "| INV-007 | Wayne Logistics | $6,200.00 | Pending | 2026-03-10 |\n\n"
                "Total: 5 invoices."
            ),
        )

    def _scenario_2(self, tool_results_so_far: int) -> Message:
        """Multi-step — find largest Acme pending invoice and approve it."""
        if tool_results_so_far == 0:
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="list_invoices",
                        arguments={"vendor": "Acme Corp", "status": "pending"},
                        call_id="call_1",
                    )
                ],
            )
        if tool_results_so_far == 1:
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="approve_invoice",
                        arguments={"invoice_id": "INV-001"},
                        call_id="call_2",
                    )
                ],
            )
        return Message(
            role="assistant",
            content=(
                "Done! I've approved INV-001 from Acme Corp for €15,000.00 "
                "(Q1 consulting services). This was the largest pending invoice "
                "from Acme Corp."
            ),
        )

    def _scenario_3(self, tool_results_so_far: int) -> Message:
        """Error handling — non-existent invoice."""
        if tool_results_so_far == 0:
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="approve_invoice",
                        arguments={"invoice_id": "INV-9999"},
                        call_id="call_1",
                    )
                ],
            )
        return Message(
            role="assistant",
            content=(
                "I wasn't able to approve invoice INV-9999 — it doesn't exist "
                "in the system. Please double-check the invoice ID and try again."
            ),
        )

    def _scenario_4(self, tool_results_so_far: int) -> Message:
        """Guardrail — bulk mutation. The mock LLM lists first, then asks."""
        if tool_results_so_far == 0:
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="list_invoices",
                        arguments={"status": "pending"},
                        call_id="call_1",
                    )
                ],
            )
        return Message(
            role="assistant",
            content=(
                "I found 5 pending invoices totaling approximately €72,900:\n\n"
                "1. INV-001 — Acme Corp — €15,000.00\n"
                "2. INV-003 — Globex Industries — €8,750.00\n"
                "3. INV-005 — Initech Solutions — €42,000.00\n"
                "4. INV-007 — Wayne Logistics — $6,200.00\n"
                "5. INV-008 — Umbrella Catering — €950.00\n\n"
                "Approving all of these is a significant action. "
                "Would you like me to proceed, or would you prefer to "
                "review and approve them individually?"
            ),
        )

    def _scenario_5(self, tool_results_so_far: int) -> Message:
        """Ambiguity — multiple Globex invoices."""
        if tool_results_so_far == 0:
            return Message(
                role="assistant",
                tool_calls=[
                    ToolCall(
                        name="list_invoices",
                        arguments={"vendor": "Globex"},
                        call_id="call_1",
                    )
                ],
            )
        return Message(
            role="assistant",
            content=(
                "I found 2 invoices from Globex Industries:\n\n"
                "1. **INV-003** — €8,750.00 — Status: Pending "
                "(Cloud hosting — February, due 2026-03-01)\n"
                "2. **INV-004** — €8,750.00 — Status: Overdue "
                "(Cloud hosting — January, due 2026-01-15)\n\n"
                "Which one were you asking about?"
            ),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _last_user_message(messages: list[Message]) -> str:
        for msg in reversed(messages):
            if msg.role == "user" and msg.content:
                return msg.content
        return ""

    @staticmethod
    def _count_tool_results(messages: list[Message]) -> int:
        return sum(1 for msg in messages if msg.role == "tool")
