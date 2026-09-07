"""npm-install the filesystem MCP server into runtime/mcp/.

Two deliberate departures from fetch_runtime.py, both because npm already does the
job that script hand-rolls:

  * No urllib download, no sha256, no resume. npm resolves the dependency tree,
    verifies each tarball against the registry's own integrity hashes and retries
    failed fetches itself. Reimplementing that here would be pure loss.
  * The whole install measured 103 packages / 14.2 s / 31 MB / 4,026 files — there
    is no long fragile transfer to make resumable in the first place.

Node discovery is app.mcp.find_node() rather than a copy of it, because that
function's third level reads the PATH out of the registry: the only level that
works in a shell whose environment predates the Node.js installer. Importing
app.mcp costs nothing — it is stdlib-only at runtime, unlike app.config.

The version is pinned to the one every measurement behind app/tools.py was taken
against: 14 tools, the annotations that tier them read-only vs writable, the
deprecated read_file alias in app.tools.FS_DROP. --version tries another.

runtime/mcp/ is covered by .gitignore's runtime/ entry, so none of the 31 MB
reaches version control.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# So `from app.mcp import ...` works when this file is run as a script, which puts
# scripts/ rather than ROOT at sys.path[0].
sys.path.insert(0, str(ROOT))

from app.mcp import (  # noqa: E402  (import after the sys.path fix above)
    FS_PACKAGE,
    FS_PACKAGE_VERSION,
    find_node,
    fs_entry_script,
)

MCP_DIR = ROOT / "runtime" / "mcp"
# One derivation shared with Settings.mcp_entry, so this script and the app can
# never disagree about where the entry script is — a disagreement would show up as
# the app saying "run fetch_mcp.py" while fetch_mcp.py says "already installed".
ENTRY = fs_entry_script(MCP_DIR)


def find_npm(node: Path) -> Path:
    """npm beside node.exe, or on PATH as a fallback.

    Beside node.exe first because that is where the installer puts it and because
    PATH is exactly the thing find_node had to work around.
    """
    beside = node.parent / ("npm.cmd" if sys.platform == "win32" else "npm")
    if beside.is_file():
        return beside
    found = shutil.which("npm")
    if found:
        return Path(found)
    raise SystemExit(f"found {node} but no npm next to it and none on PATH")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default=FS_PACKAGE_VERSION,
        help=f"package version to install (default {FS_PACKAGE_VERSION})",
    )
    parser.add_argument(
        "--node", default="", help="path to node.exe (default: auto-discover)"
    )
    parser.add_argument("--force", action="store_true", help="reinstall even if present")
    args = parser.parse_args()

    node = find_node(args.node)
    if node is None:
        raise SystemExit(
            "node.exe not found. Install Node.js from https://nodejs.org and open a\n"
            "new shell (an existing one keeps the PATH it was started with), or pass\n"
            "--node. MCP is optional: the app runs without it, with the 文件 and\n"
            "写入 checkboxes greyed out."
        )
    print(f"node: {node}", flush=True)

    if ENTRY.is_file():
        if not args.force:
            print(f"{ENTRY} already exists, use --force to reinstall", flush=True)
            return 0
        # npm would prune and rewrite in place, but a half-removed tree from an
        # interrupted earlier run is what --force is most often used to repair.
        shutil.rmtree(MCP_DIR, ignore_errors=True)
        print(f"removed {MCP_DIR}", flush=True)

    npm = find_npm(node)
    MCP_DIR.mkdir(parents=True, exist_ok=True)
    spec = f"{FS_PACKAGE}@{args.version}"
    print(f"installing {spec} into {MCP_DIR}", flush=True)
    started = time.time()
    # stdout and stderr inherited on purpose: npm's own progress output is the only
    # feedback during a ~14 s install, and capturing it would look like a hang.
    # --prefix plus cwd is belt and braces — cwd alone is enough for npm to write
    # package.json and package-lock.json here, --prefix pins the node_modules
    # location regardless of any parent package.json it might otherwise walk up to.
    done = subprocess.run(
        [str(npm), "install", "--prefix", str(MCP_DIR), "--no-audit", "--no-fund", spec],
        cwd=str(MCP_DIR),
    )
    if done.returncode != 0:
        raise SystemExit(f"npm install failed with exit code {done.returncode}")
    print(f"installed in {time.time() - started:.1f} s", flush=True)

    # Mirrors fetch_runtime.py's "llama-server.exe not found after extraction": npm
    # can exit 0 having laid out something other than what Settings.mcp_entry
    # derives, and finding that out at the first tool call would be far more
    # confusing than finding it out here.
    if not ENTRY.is_file():
        raise SystemExit(f"{ENTRY} not found after install")
    print(f"entry: {ENTRY}", flush=True)
    print("restart the app to pick it up (run.py has no auto-reload).", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
