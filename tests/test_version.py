"""The version the About panel shows: README as its only source, and the sync that
follows from it.

Standard library only, like every other test here. Run from the project root:

    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe -m unittest tests.test_version -v

The point of the cross-file cases is that `app/version.py` turned a documentation
convention into a runtime contract. The version a release announces is written out
by hand in several places (README's version line, its 版本历史 entry, the dev-log
index, the record's own filename), so the agreement between them is exactly the kind
of thing a test should own. Read them the way the app does, by pattern, not by line
number -- the surrounding prose grows with every release.
"""

from __future__ import annotations

import os
import re
import unittest
from pathlib import Path

from app.version import ROOT, README, read_version

# Same shape app/version.py looks for, kept separate on purpose: if the two patterns
# ever disagree, this file fails rather than quietly agreeing with itself.
README_VERSION_RE = re.compile(r"^\*\*版本 / Version\*\*: v(\d+\.\d+\.\d+)", re.MULTILINE)
HISTORY_ENTRY_RE = re.compile(r"^### v(\d+\.\d+\.\d+)", re.MULTILINE)
RECORD_NAME_RE = re.compile(r"^v(\d+\.\d+\.\d+)_\d{4}-\d{2}-\d{2}_", re.ASCII)
INDEX_ENTRY_RE = re.compile(r"^- v(\d+\.\d+\.\d+) ", re.MULTILINE)

DEVLOG = ROOT / "docs" / "开发记录"


def _key(text: str, pattern: re.Pattern[str]) -> str | None:
    """First version a pattern matches, or None."""
    match = pattern.search(text)
    return match.group(1) if match else None


class ReadFromReadme(unittest.TestCase):
    def test_readme_publishes_a_parseable_version(self):
        version = read_version()
        self.assertIsNotNone(version, "README.md has no line matching '**版本 / Version**: vX.Y.Z'")
        self.assertRegex(version, r"^\d+\.\d+\.\d+$")

    def test_reads_the_repository_readme_not_a_copy(self):
        # The default argument and README.resolve() must agree, or a stale README under
        # a different working directory would be served to the panel.
        self.assertEqual(read_version(), read_version(README))
        self.assertEqual(read_version(README), _key(README.read_text(encoding="utf-8"), README_VERSION_RE))

    def test_missing_file_answers_none_and_warns(self):
        missing = ROOT / "runtime" / "definitely-not-a-readme.md"
        self.assertFalse(missing.exists())
        with self.assertLogs("app.version", level="WARNING") as logs:
            self.assertIsNone(read_version(missing))
        self.assertIn("definitely-not-a-readme", "\n".join(logs.output))

    def test_unrecognised_shape_answers_none(self):
        # The failure mode this guards is a hand-edit of the line: re-labelled, full
        # width colon, or a version without the patch component all stop matching, and
        # the panel must then say 未知 rather than serve a stale cached value.
        for bad in (
            "**版本 / Version**：v1.2.0",
            "**Version**: v1.2",
            "版本 v1.2.0",
        ):
            with self.subTest(line=bad), self.assertLogs("app.version", level="WARNING"):
                self.assertIsNone(read_version(self._write(bad)))

    def _write(self, body: str) -> Path:
        path = ROOT / "runtime" / "test-version-readme.md"
        path.write_text(f"# t\n\n{body}\n", encoding="utf-8")
        self.addCleanup(path.unlink, missing_ok=True)
        return path

    def test_cache_follows_an_edit_without_a_restart(self):
        path = self._write("**版本 / Version**: v1.0.0")
        self.assertEqual(read_version(path), "1.0.0")
        path.write_text("**版本 / Version**: v1.1.0\n", encoding="utf-8")
        # An in-place rewrite can land inside the same mtime tick the previous stat
        # recorded, so pin the timestamps rather than trusting the clock.
        os.utime(path, ns=(9_000_000_000, 9_000_000_000))
        self.assertEqual(read_version(path), "1.1.0")


class ReleaseStaysSynchronised(unittest.TestCase):
    """The hand-written places a release touches must tell one version."""

    def test_history_section_leads_with_the_released_version(self):
        version = read_version()
        text = README.read_text(encoding="utf-8")
        section = text.split("## 9. 版本历史", 1)
        self.assertEqual(len(section), 2, "README.md lost its 版本历史 section")
        self.assertEqual(_key(section[1], HISTORY_ENTRY_RE), version)

    def test_devlog_index_leads_with_the_released_version(self):
        index = DEVLOG / "index.md"
        self.assertTrue(index.is_file(), "docs/开发记录/index.md is missing")
        body = index.read_text(encoding="utf-8").split("## 记录列表", 1)[-1]
        self.assertEqual(_key(body, INDEX_ENTRY_RE), read_version())

    def test_newest_record_matches_the_released_version(self):
        records = sorted(
            (m.group(1) for p in DEVLOG.glob("v*_*.md") if (m := RECORD_NAME_RE.match(p.name))),
            key=lambda v: tuple(int(n) for n in v.split(".")),
        )
        if not records:
            # v1.0.0 only established the scheme and was summarised in README.md instead
            # of getting its own record. Once any record exists, the newest one is the
            # release the README advertises.
            self.skipTest("no iteration record exists yet")
        self.assertEqual(records[-1], read_version())

    def test_record_names_are_well_formed(self):
        for path in DEVLOG.glob("v*.md"):
            self.assertIsNotNone(
                RECORD_NAME_RE.match(path.name),
                f"{path.name} is not named vX.Y.Z_YYYY-MM-DD_描述.md",
            )


class WireContract(unittest.TestCase):
    def test_about_endpoint_reports_the_version_field(self):
        # main.py must keep sending the key, not merely send it when a value exists:
        # the panel distinguishes "未知" from "字段不存在" by presence alone.
        source = (ROOT / "app" / "main.py").read_text(encoding="utf-8")
        about = source.split("@app.get(\"/api/about\")", 1)[-1].split("@app.", 1)[0]
        self.assertIn('"version": read_version()', about)

    def test_frontend_keeps_the_v_prefix_out_of_the_payload(self):
        script = (ROOT / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("a.version ? `v${a.version}`", script)


if __name__ == "__main__":
    unittest.main()
