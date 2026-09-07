"""The server-side half of the empty-turn filter: main._is_sendable().

The client half (isSendable / wireMessages in static/app.js) has no unit test: app.js
is a browser classic script whose neighbours touch the DOM, and this repo has no JS
test harness — adding one would mean a new dependency. It is verified in the browser
console instead, which works because a classic script makes both real globals.
"""

from __future__ import annotations

import unittest

from app.main import DocumentIn, MessageIn, _is_sendable

PNG = "data:image/png;base64,AAAA"


class TestAssistantTurns(unittest.TestCase):
    def test_empty_content_dropped(self):
        self.assertFalse(_is_sendable(MessageIn(role="assistant", content="")))

    def test_whitespace_content_dropped(self):
        self.assertFalse(_is_sendable(MessageIn(role="assistant", content="   \n ")))

    def test_nonempty_content_kept(self):
        self.assertTrue(_is_sendable(MessageIn(role="assistant", content="你好，我在。")))


class TestUserTurns(unittest.TestCase):
    def test_blank_content_with_documents_kept(self):
        m = MessageIn(role="user", content="", documents=[DocumentIn(name="a.md", text="正文")])
        self.assertTrue(_is_sendable(m))

    def test_blank_content_with_images_kept(self):
        self.assertTrue(_is_sendable(MessageIn(role="user", content="", images=[PNG])))

    def test_blank_with_nothing_dropped(self):
        self.assertFalse(_is_sendable(MessageIn(role="user", content="  ")))


class TestTheReportedFailure(unittest.TestCase):
    def test_degenerate_replay_shape_is_broken_up(self):
        """The exact shape found in the session that reported 罢工.

        msgs[6] and msgs[8] were byte-identical user text and msgs[7] and msgs[9] were
        both empty assistant turns, so the model was handed
        […, user X, assistant "", user X] and answered with an immediate EOS
        (GENERATED 1 tokens, ctx=5972, truncated=0 — 4.5% of the window).
        """
        msgs = [
            MessageIn(role="user", content="帮我写一首诗"),
            MessageIn(role="assistant", content=""),
            MessageIn(role="user", content="帮我写一首诗"),
        ]
        kept = [m for m in msgs if _is_sendable(m)]
        self.assertEqual([m.role for m in kept], ["user", "user"])

    def test_error_turns_are_not_the_servers_problem(self):
        """Documented asymmetry, asserted so it cannot drift silently.

        `error` is not a field on MessageIn, so a turn whose content is our own
        "请求失败：…" text is indistinguishable from a real answer here and is kept.
        app.js drops it client-side, where msg.error still exists after a restore.
        """
        self.assertNotIn("error", MessageIn.model_fields)
        self.assertTrue(_is_sendable(MessageIn(role="assistant", content="请求失败：网络中断")))


if __name__ == "__main__":
    unittest.main()
