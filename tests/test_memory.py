"""The native memory store: one JSON file of short notes, driven against a temp dir.

Standard library only, by decision — pytest was explicitly rejected as a new
dependency. Run from the project root:

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m unittest discover -s tests -v

memory_store.py takes plain Paths and imports nothing from the package, so a
TemporaryDirectory is the whole fixture — the same shape history.py's docstring
describes for itself.

Two behaviours here are easy to get wrong and expensive to notice later, so they get
their own tests. `list_memories` has to be newest-first even when every record shares a
timestamp, which is all of them in a test run, because `created` has second
granularity; a plain sort leaves them in file order and the panel shows the oldest
note on top. And `search_memories` is keyword scoring with CJK bigrams, not semantic
retrieval: the tests pin what it does find so that what it does not find stays honest
in the UI and in .env.example.
"""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path

from app.memory_store import (
    DEFAULT_KIND,
    KINDS,
    MAX_MEMORIES,
    MAX_MEMORY_CHARS,
    MEMORY_ID_RE,
    MEMORY_NAME,
    MemoryStoreError,
    _atomic_write_json,
    _read_store,
    _terms,
    add_memory,
    clear_memories,
    delete_memory,
    is_memory_id,
    list_memories,
    new_memory_id,
    search_memories,
)


class StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.path = self.dir / "runtime" / MEMORY_NAME

    def texts(self, records):
        return [r["text"] for r in records]


class TestIds(StoreCase):
    def test_new_ids_are_well_formed_and_unique(self):
        seen = {new_memory_id() for _ in range(200)}
        self.assertEqual(len(seen), 200)
        for value in seen:
            self.assertTrue(is_memory_id(value), value)

    def test_malformed_ids_are_rejected(self):
        for value in ("", "abc", "0" * 31, "0" * 33, "0" * 32 + "x", "Z" * 32,
                      "../" + "0" * 32, "0" * 16 + "-" + "0" * 15):
            with self.subTest(repr(value)):
                self.assertFalse(is_memory_id(value))

    def test_uppercase_hex_is_rejected(self):
        """uuid4().hex is lowercase, so an uppercase id did not come from here.
        delete_memory's id arrives from the browser and is checked before anything
        else — same rule as history.SESSION_ID_RE."""
        self.assertFalse(is_memory_id(new_memory_id().upper()))

    def test_none_is_rejected_rather_than_raising(self):
        self.assertFalse(is_memory_id(None))

    def test_the_regex_is_the_documented_shape(self):
        self.assertEqual(MEMORY_ID_RE.pattern, "^[0-9a-f]{32}$")


class TestAdd(StoreCase):
    def test_returns_a_complete_record(self):
        record = add_memory(self.path, "用户偏好深色主题", "preference")
        self.assertEqual(set(record), {"id", "kind", "text", "created"})
        self.assertTrue(is_memory_id(record["id"]))
        self.assertEqual(record["kind"], "preference")
        self.assertEqual(record["text"], "用户偏好深色主题")
        self.assertRegex(record["created"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}")

    def test_creates_the_parent_directory(self):
        self.assertFalse(self.path.parent.exists())
        add_memory(self.path, "x")
        self.assertTrue(self.path.is_file())

    def test_strips_surrounding_whitespace(self):
        self.assertEqual(add_memory(self.path, "  内容 \n ")["text"], "内容")

    def test_blank_text_is_refused(self):
        for value in ("", "   ", "\n\t ", None):
            with self.subTest(repr(value)):
                with self.assertRaises(MemoryStoreError) as caught:
                    add_memory(self.path, value)
                self.assertIn("空的", str(caught.exception))

    def test_a_refused_memory_writes_nothing(self):
        """An unwritten note must never leave a file behind that says otherwise."""
        with self.assertRaises(MemoryStoreError):
            add_memory(self.path, "")
        self.assertFalse(self.path.exists())

    def test_over_long_text_is_refused_with_its_length(self):
        body = "长" * (MAX_MEMORY_CHARS + 1)
        with self.assertRaises(MemoryStoreError) as caught:
            add_memory(self.path, body)
        message = str(caught.exception)
        self.assertIn(str(MAX_MEMORY_CHARS + 1), message)
        self.assertIn(str(MAX_MEMORY_CHARS), message)

    def test_exactly_at_the_length_cap_is_accepted(self):
        self.assertEqual(len(add_memory(self.path, "长" * MAX_MEMORY_CHARS)["text"]),
                         MAX_MEMORY_CHARS)

    def test_an_unknown_kind_degrades_to_the_default(self):
        """The remember tool exposes kind as an enum and an 8B model still sends "note"
        now and then; losing the memory over a label is the wrong trade."""
        for value in ("note", "info", "PREFERENCE", "", None):
            with self.subTest(repr(value)):
                self.assertEqual(add_memory(self.path, f"x{value}", value)["kind"], DEFAULT_KIND)

    def test_every_declared_kind_is_accepted(self):
        for kind in KINDS:
            with self.subTest(kind):
                self.assertEqual(add_memory(self.path, f"关于{kind}", kind)["kind"], kind)

    def test_an_exact_duplicate_collapses(self):
        first = add_memory(self.path, "用户偏好深色主题")
        second = add_memory(self.path, "用户偏好深色主题")
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(len(_read_store(self.path)), 1)

    def test_a_duplicate_collapses_ignoring_case_and_whitespace(self):
        add_memory(self.path, "Export as PDF")
        self.assertEqual(len(_read_store(self.path)), 1)
        add_memory(self.path, "  export as pdf \n")
        self.assertEqual(len(_read_store(self.path)), 1)

    def test_a_near_duplicate_is_kept(self):
        """Collapsing is exact-match only: quietly dropping a note that differs by one
        word would lose information the model was told it had saved."""
        add_memory(self.path, "用户偏好深色主题")
        add_memory(self.path, "用户偏好浅色主题")
        self.assertEqual(len(_read_store(self.path)), 2)

    def test_the_cap_refuses_rather_than_evicting(self):
        """Silent eviction means the model believes something is remembered when it is
        not, and the user has no way to notice."""
        _atomic_write_json(self.path, [
            {"id": new_memory_id(), "kind": "fact", "text": f"第 {i} 条", "created": "2026-01-01T00:00:00+00:00"}
            for i in range(MAX_MEMORIES)
        ])
        with self.assertRaises(MemoryStoreError) as caught:
            add_memory(self.path, "新的一条")
        message = str(caught.exception)
        self.assertIn(str(MAX_MEMORIES), message)
        self.assertIn("记忆」面板", message)
        self.assertEqual(len(_read_store(self.path)), MAX_MEMORIES)

    def test_one_slot_below_the_cap_still_accepts(self):
        _atomic_write_json(self.path, [
            {"id": new_memory_id(), "kind": "fact", "text": f"第 {i} 条", "created": "2026-01-01T00:00:00+00:00"}
            for i in range(MAX_MEMORIES - 1)
        ])
        add_memory(self.path, "新的一条")
        self.assertEqual(len(_read_store(self.path)), MAX_MEMORIES)

    def test_a_duplicate_at_the_cap_still_returns_the_existing_note(self):
        """The duplicate check runs first, so recall keeps working on a full store."""
        _atomic_write_json(self.path, [
            {"id": new_memory_id(), "kind": "fact", "text": f"第 {i} 条", "created": "2026-01-01T00:00:00+00:00"}
            for i in range(MAX_MEMORIES)
        ])
        self.assertEqual(add_memory(self.path, "第 0 条")["text"], "第 0 条")

    def test_no_temp_file_is_left_behind(self):
        add_memory(self.path, "x")
        add_memory(self.path, "y")
        self.assertEqual([p.name for p in self.path.parent.iterdir()], [MEMORY_NAME])

    def test_the_file_is_human_readable_json(self):
        """The panel is the audit trail, but a user should also be able to open the file."""
        add_memory(self.path, "用户偏好深色主题", "preference")
        raw = self.path.read_text(encoding="utf-8")
        self.assertIn("用户偏好深色主题", raw)
        self.assertEqual(json.loads(raw)[0]["kind"], "preference")


class TestReadDegradation(StoreCase):
    """Reads swallow errors, writes raise — the asymmetry history.py documents."""

    def test_a_missing_file_is_an_empty_store(self):
        self.assertEqual(list_memories(self.path), [])
        self.assertEqual(search_memories(self.path, "任何"), [])
        self.assertEqual(_read_store(self.path), [])

    def test_a_corrupt_file_is_an_empty_store(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{ not json", encoding="utf-8")
        self.assertEqual(list_memories(self.path), [])

    def test_a_json_object_instead_of_a_list_is_an_empty_store(self):
        _atomic_write_json(self.path, {"memories": []})
        self.assertEqual(list_memories(self.path), [])

    def test_malformed_records_are_dropped_and_the_rest_kept(self):
        """One hand-edited line must not cost the user every other memory."""
        good = {"id": new_memory_id(), "kind": "fact", "text": "好的一条", "created": "2026-01-01T00:00:00+00:00"}
        _atomic_write_json(self.path, [
            good,
            {"id": "not-an-id", "text": "坏 id", "created": "2026-01-01T00:00:00+00:00"},
            {"id": new_memory_id(), "text": "", "created": "2026-01-01T00:00:00+00:00"},
            {"id": new_memory_id(), "text": "   ", "created": "2026-01-01T00:00:00+00:00"},
            "a string",
            7,
            None,
            {"id": new_memory_id(), "text": "没有 created"},
        ])
        self.assertEqual(self.texts(list_memories(self.path)), ["好的一条", "没有 created"])

    def test_a_directory_in_place_of_the_file_is_an_empty_store(self):
        self.path.mkdir(parents=True)
        self.assertEqual(list_memories(self.path), [])


class TestListOrder(StoreCase):
    def test_newest_first(self):
        for stamp in ("2026-01-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00",
                      "2026-03-01T00:00:00+00:00"):
            _atomic_write_json(self.path, [
                *(_read_store(self.path)),
                {"id": new_memory_id(), "kind": "fact", "text": stamp[:7], "created": stamp},
            ])
        self.assertEqual(self.texts(list_memories(self.path)), ["2026-06", "2026-03", "2026-01"])

    def test_records_added_in_the_same_second_stay_newest_first(self):
        """The pin behind reversing before sorting. `created` has second granularity,
        so every record a test run adds shares a timestamp; a plain sort leaves them in
        file order and the panel shows the oldest note on top."""
        for text in ("第一条", "第二条", "第三条"):
            add_memory(self.path, text)
        stored = _read_store(self.path)
        self.assertEqual(len({r["created"] for r in stored}), 1)
        self.assertEqual(self.texts(list_memories(self.path)), ["第三条", "第二条", "第一条"])

    def test_a_hand_reordered_file_is_still_sorted(self):
        _atomic_write_json(self.path, [
            {"id": new_memory_id(), "kind": "fact", "text": "新", "created": "2026-06-01T00:00:00+00:00"},
            {"id": new_memory_id(), "kind": "fact", "text": "旧", "created": "2026-01-01T00:00:00+00:00"},
        ])
        self.assertEqual(self.texts(list_memories(self.path)), ["新", "旧"])

    def test_listing_does_not_mutate_the_file(self):
        add_memory(self.path, "x")
        before = self.path.read_bytes()
        list_memories(self.path)
        search_memories(self.path, "x")
        self.assertEqual(self.path.read_bytes(), before)


class TestTerms(unittest.TestCase):
    def test_latin_words(self):
        self.assertEqual(sorted(_terms("PDF export")), ["export", "pdf"])

    def test_latin_is_lowercased_and_needs_two_characters(self):
        self.assertEqual(sorted(_terms("A bb C_c")), ["bb", "c_c"])

    def test_cjk_is_split_into_bigrams(self):
        """Chinese has no spaces, so whitespace splitting would hand 导出格式 to the
        matcher as one token and match nothing."""
        self.assertEqual(_terms("导出格式"), ["导出", "出格", "格式"])

    def test_a_single_cjk_character_is_kept_whole(self):
        self.assertEqual(_terms("好"), ["好"])

    def test_mixed_text(self):
        self.assertEqual(sorted(_terms("导出 PDF")), sorted(["pdf", "导出"]))

    def test_punctuation_only_yields_nothing(self):
        self.assertEqual(_terms("？！。，"), [])


class TestSearch(StoreCase):
    def seed(self):
        add_memory(self.path, "导出格式默认用 PDF", "preference")
        add_memory(self.path, "用户偏好深色主题", "preference")
        add_memory(self.path, "把测试跑完再提交", "todo")
        add_memory(self.path, "项目用 FastAPI 加 llama.cpp", "fact")

    def test_a_blank_query_returns_nothing(self):
        """Not the whole store: recall answers a question, and dumping 500 notes into
        the context is the failure this module exists to avoid."""
        self.seed()
        for value in ("", "   ", None, "？！"):
            with self.subTest(repr(value)):
                self.assertEqual(search_memories(self.path, value), [])

    def test_finds_a_cjk_bigram(self):
        self.seed()
        self.assertIn("导出格式默认用 PDF", self.texts(search_memories(self.path, "导出")))

    def test_finds_a_latin_word_case_insensitively(self):
        self.seed()
        self.assertIn("导出格式默认用 PDF", self.texts(search_memories(self.path, "pdf")))
        self.assertIn("项目用 FastAPI 加 llama.cpp",
                      self.texts(search_memories(self.path, "FASTAPI")))

    def test_a_multi_word_query_scores_the_stronger_match_first(self):
        """导出格式 深色 → bigrams 导出/出格/格式/深色. The export memory matches three of
        them (6 points), the theme memory one (2). Written with 格式 in the query on
        purpose: "导出 深色" alone is a tie, because each side then matches exactly one
        2-character bigram, and the test would be asserting an ordering the scorer never
        produced."""
        self.seed()
        hits = self.texts(search_memories(self.path, "导出格式 深色"))
        self.assertEqual(hits[0], "导出格式默认用 PDF")
        self.assertIn("用户偏好深色主题", hits)

    def test_an_equal_score_falls_back_to_newest_first(self):
        """The tie-break, pinned because it is the honest limit of keyword scoring: two
        memories that match equally well are ordered by when they were saved, not by
        which one is more relevant."""
        self.seed()
        hits = self.texts(search_memories(self.path, "导出 深色"))
        self.assertEqual(sorted(hits), sorted(["导出格式默认用 PDF", "用户偏好深色主题"]))
        self.assertEqual(hits[0], "用户偏好深色主题", "the later note wins a tie")

    def test_a_longer_match_outweighs_an_incidental_bigram(self):
        """Length as weight, so a real term beats a bigram that happens to appear."""
        add_memory(self.path, "python 项目")
        add_memory(self.path, "thon 只是碰巧出现")
        self.assertEqual(self.texts(search_memories(self.path, "python"))[0], "python 项目")

    def test_a_kind_match_adds_weight(self):
        """The kind is searchable on its own: "todo" appears in no text here, so the only
        way the third note can come back is the +1 it scores against its own kind."""
        add_memory(self.path, "第一条无关内容")
        add_memory(self.path, "第二条也无关")
        add_memory(self.path, "记得把附件删掉", "todo")
        hits = search_memories(self.path, "todo")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["kind"], "todo")

    def test_no_hits_is_an_empty_list_not_an_error(self):
        self.seed()
        self.assertEqual(search_memories(self.path, "量子纠缠"), [])

    def test_respects_the_limit(self):
        for index in range(12):
            add_memory(self.path, f"关于导出的第 {index} 条")
        self.assertEqual(len(search_memories(self.path, "导出", 3)), 3)
        self.assertEqual(len(search_memories(self.path, "导出", 50)), 12)

    def test_the_default_limit_is_what_recall_uses(self):
        for index in range(12):
            add_memory(self.path, f"关于导出的第 {index} 条")
        self.assertEqual(len(search_memories(self.path, "导出")), 5)

    def test_a_limit_below_one_still_returns_one(self):
        add_memory(self.path, "关于导出的一条")
        self.assertEqual(len(search_memories(self.path, "导出", 0)), 1)

    def test_ties_come_out_newest_first(self):
        for text in ("导出甲", "导出乙", "导出丙"):
            add_memory(self.path, text)
        self.assertEqual(self.texts(search_memories(self.path, "导出")),
                         ["导出丙", "导出乙", "导出甲"])

    def test_malformed_records_are_ignored_by_search_too(self):
        _atomic_write_json(self.path, [
            "junk",
            {"id": new_memory_id(), "kind": "fact", "text": "好的一条 导出", "created": "2026-01-01T00:00:00+00:00"},
        ])
        self.assertEqual(self.texts(search_memories(self.path, "导出")), ["好的一条 导出"])

    def test_searching_an_empty_store(self):
        self.assertEqual(search_memories(self.path, "导出"), [])


class TestDelete(StoreCase):
    def test_removes_one_and_reports_true(self):
        kept = add_memory(self.path, "留下的")
        gone = add_memory(self.path, "删掉的")
        self.assertTrue(delete_memory(self.path, gone["id"]))
        self.assertEqual(self.texts(_read_store(self.path)), ["留下的"])
        self.assertIn(kept["id"], [r["id"] for r in _read_store(self.path)])

    def test_a_malformed_id_is_refused_outright(self):
        add_memory(self.path, "x")
        for value in ("", "abc", "../x", "0" * 31, new_memory_id().upper(), None):
            with self.subTest(repr(value)):
                with self.assertRaises(MemoryStoreError):
                    delete_memory(self.path, value)
        self.assertEqual(len(_read_store(self.path)), 1)

    def test_an_unknown_but_well_formed_id_reports_false(self):
        """False rather than an error: the panel deletes by id and a second click on a
        row that is already gone is not a failure worth an alert."""
        add_memory(self.path, "x")
        self.assertFalse(delete_memory(self.path, new_memory_id()))
        self.assertEqual(len(_read_store(self.path)), 1)

    def test_deleting_from_an_empty_store_reports_false(self):
        self.assertFalse(delete_memory(self.path, new_memory_id()))

    def test_deleting_the_last_memory_leaves_a_valid_empty_file(self):
        gone = add_memory(self.path, "唯一一条")
        delete_memory(self.path, gone["id"])
        self.assertEqual(json.loads(self.path.read_text(encoding="utf-8")), [])
        self.assertEqual(list_memories(self.path), [])

    def test_no_temp_file_is_left_behind(self):
        record = add_memory(self.path, "x")
        delete_memory(self.path, record["id"])
        self.assertEqual([p.name for p in self.path.parent.iterdir()], [MEMORY_NAME])


class TestClear(StoreCase):
    def test_returns_the_count_and_removes_the_file(self):
        for index in range(3):
            add_memory(self.path, f"第 {index} 条")
        self.assertEqual(clear_memories(self.path), 3)
        self.assertFalse(self.path.exists())
        self.assertEqual(list_memories(self.path), [])

    def test_an_empty_store_clears_to_zero(self):
        self.assertEqual(clear_memories(self.path), 0)

    def test_clearing_twice_is_not_an_error(self):
        add_memory(self.path, "x")
        self.assertEqual(clear_memories(self.path), 1)
        self.assertEqual(clear_memories(self.path), 0)

    def test_the_store_is_usable_afterwards(self):
        add_memory(self.path, "旧的")
        clear_memories(self.path)
        self.assertEqual(add_memory(self.path, "新的")["text"], "新的")

    def test_malformed_records_are_not_counted(self):
        _atomic_write_json(self.path, ["junk", 7])
        self.assertEqual(clear_memories(self.path), 0)


class TestConcurrency(StoreCase):
    def test_parallel_writes_lose_nothing(self):
        """_LOCK serialises every read-modify-write. run.py starts uvicorn without
        workers, so one process holds one lock and that is enough — but ToolRunner
        reaches the store through asyncio.to_thread, so the threads are real."""
        threads = 8
        per_thread = 10
        errors = []

        def worker(index):
            try:
                for step in range(per_thread):
                    add_memory(self.path, f"线程 {index} 的第 {step} 条")
            except Exception as exc:  # noqa: BLE001 - reported to the assertion below
                errors.append(exc)

        started = [threading.Thread(target=worker, args=(i,)) for i in range(threads)]
        for thread in started:
            thread.start()
        for thread in started:
            thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(len(_read_store(self.path)), threads * per_thread)
        self.assertEqual(len({r["id"] for r in _read_store(self.path)}), threads * per_thread)

    def test_parallel_deletes_do_not_corrupt_the_file(self):
        records = [add_memory(self.path, f"第 {index} 条") for index in range(20)]
        threads = [threading.Thread(target=delete_memory, args=(self.path, r["id"]))
                   for r in records]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(_read_store(self.path), [])


if __name__ == "__main__":
    unittest.main()
