"""Budget layer: budget_safety(), fit_budget()'s contract, and the image charge.

Standard library only, by decision — pytest was explicitly rejected as a new
dependency. Run from the project root:

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m unittest discover -s tests -v
"""

from __future__ import annotations

import inspect
import json
import unittest

from app.config import DEFAULT_SYSTEM_PROMPT
from app.documents import (
    CTX_SAFETY_TOKENS,
    PER_MESSAGE_TEMPLATE_TOKENS,
    TEMPLATE_SLACK_TOKENS,
    TOKENS_PER_IMAGE,
    _msg_tokens,
    budget_safety,
    estimate_tokens,
    fit_budget,
)
from app.llm import ChatMessage
from app.main import MAX_IMAGES_PER_MESSAGE
from app.search import TOOLS

TOOLS_JSON = json.dumps(TOOLS, ensure_ascii=False)


class TestBudgetSafety(unittest.TestCase):
    def test_without_tools(self):
        self.assertEqual(
            budget_safety(DEFAULT_SYSTEM_PROMPT, "", 0),
            estimate_tokens(DEFAULT_SYSTEM_PROMPT) + TEMPLATE_SLACK_TOKENS,
        )

    def test_adds_exactly_the_tools_cost(self):
        self.assertEqual(
            budget_safety(DEFAULT_SYSTEM_PROMPT, TOOLS_JSON, 0)
            - budget_safety(DEFAULT_SYSTEM_PROMPT, "", 0),
            estimate_tokens(TOOLS_JSON),
        )

    def test_scales_per_message(self):
        for n in (0, 1, 10, 200):
            with self.subTest(n=n):
                self.assertEqual(
                    budget_safety(DEFAULT_SYSTEM_PROMPT, "", n + 1)
                    - budget_safety(DEFAULT_SYSTEM_PROMPT, "", n),
                    PER_MESSAGE_TEMPLATE_TOKENS,
                )

    def test_real_overhead_exceeds_the_old_hardcoded_margin(self):
        """The bug this change fixes, pinned as a regression test.

        The system prompt and the tool schemas cost more between them than the flat
        CTX_SAFETY_TOKENS = 1024 allowance — before a single message or a template
        token is counted. Measured: 354 + 807 = 1161.
        """
        self.assertGreater(
            estimate_tokens(DEFAULT_SYSTEM_PROMPT) + estimate_tokens(TOOLS_JSON),
            CTX_SAFETY_TOKENS,
        )
        self.assertGreater(budget_safety(DEFAULT_SYSTEM_PROMPT, TOOLS_JSON, 10), CTX_SAFETY_TOKENS)


class TestFitBudgetContract(unittest.TestCase):
    def test_signature_unchanged(self):
        """budget_safety() is passed in by the caller; the default stays for anyone
        with nothing to measure, and documents.py must not grow a dependency on the
        rest of the package to compute it itself."""
        params = list(inspect.signature(fit_budget).parameters.values())
        self.assertEqual([p.name for p in params], ["messages", "n_ctx", "reserve", "safety"])
        self.assertEqual(params[3].default, CTX_SAFETY_TOKENS)

    def test_larger_safety_drops_more(self):
        # estimate_tokens("一" * 100) == 101, so four of them total 404.
        msgs = [ChatMessage(role="user", content="一" * 100) for _ in range(4)]

        fitted, notes = fit_budget(msgs, 10_000, 1_000, safety=0)
        self.assertEqual(len(fitted), 4)
        self.assertEqual(notes, [])

        # budget = 10_000 - 1_000 - 8_750 = 250, so Stage C pops until two remain.
        fitted, notes = fit_budget(msgs, 10_000, 1_000, safety=8_750)
        self.assertEqual(len(fitted), 2)
        self.assertTrue(notes)

    def test_empty_list_returns_no_notes(self):
        """Why chat() overrides notes when every turn was filtered out: with no notes,
        the `if not fitted` branch would report a context overflow that never happened."""
        self.assertEqual(fit_budget([], 131_072, 4_096), ([], []))


class TestImageCharge(unittest.TestCase):
    def test_charged_per_image(self):
        base = ChatMessage(role="user", content="看看这些图")
        with_images = ChatMessage(
            role="user",
            content="看看这些图",
            images=["data:image/png;base64,AAAA"] * 2,
        )
        self.assertEqual(_msg_tokens(with_images) - _msg_tokens(base), 2 * TOKENS_PER_IMAGE)

    def test_four_images_survive_the_smallest_window(self):
        """The image charge must not create a hard failure of its own.

        Worst case: the smallest configured window (the N_CTX=16384 fallback), the
        thinking reserve, the largest possible safety (search on, MAX_CHAT_MESSAGES
        turns) and a full complement of images on one turn.
        """
        msgs = [
            ChatMessage(
                role="user",
                content="这四张图里有什么？",
                images=["data:image/png;base64,AAAA"] * MAX_IMAGES_PER_MESSAGE,
            )
        ]
        safety = budget_safety(DEFAULT_SYSTEM_PROMPT, TOOLS_JSON, 200)
        fitted, notes = fit_budget(msgs, 16_384, 4_096, safety)
        self.assertEqual(len(fitted), 1)
        self.assertEqual(notes, [])


if __name__ == "__main__":
    unittest.main()
