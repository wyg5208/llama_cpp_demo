"""Prepare a prebuilt llama.cpp runtime: resumable download, sha256 verify, extract.

GitHub release downloads are slow and stall-prone on some networks, so every
attempt resumes from the bytes already on disk instead of restarting.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

API_RELEASES = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
USER_AGENT = "llama-cpp-demo-setup/1.0"
ROOT = Path(__file__).resolve().parent.parent
RUNTIME_DIR = ROOT / "runtime"
DOWNLOAD_DIR = RUNTIME_DIR / "_downloads"

ASSET_PATTERNS = {
    "cuda": re.compile(r"^llama-b\d+-bin-win-cuda-13\.\d+-x64\.zip$"),
    "vulkan": re.compile(r"^llama-b\d+-bin-win-vulkan-x64\.zip$"),
    "cpu": re.compile(r"^llama-b\d+-bin-win-cpu-x64\.zip$"),
}

# llama.cpp's CUDA build links against these; they ship with the CUDA toolkit or
# the 391MB cudart redistributable, but are also bundled by Ollama and by any
# cu13 PyTorch install, which is far cheaper than downloading them again.
CUDA_LIB_NAMES = ("cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll")
CUDA_LIB_SEARCH_ROOTS = (
    Path.home() / "AppData/Local/Programs/Ollama/lib/ollama/cuda_v13",
    Path("E:/Programs/ComfyUI-aki-v20/python/Lib/site-packages/torch/lib"),
)


def http_get_json(url: str, attempts: int = 8) -> object:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
            return json.loads(body)
        except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError) as exc:
            last = exc
            print(f"  api attempt {attempt}: {type(exc).__name__}, retrying", flush=True)
            time.sleep(min(2 * attempt, 10))
    raise SystemExit(f"could not fetch {url}: {last}")


def resolve_asset(backend: str) -> dict:
    """Find the newest nightly build that publishes an asset for this backend."""
    pattern = ASSET_PATTERNS[backend]
    releases = http_get_json(f"{API_RELEASES}?per_page=20")
    for release in releases:
        tag = release.get("tag_name", "")
        if not re.fullmatch(r"b\d+", tag):
            continue
        for asset in release.get("assets", []):
            if pattern.match(asset["name"]):
                return {
                    "tag": tag,
                    "name": asset["name"],
                    "url": asset["browser_download_url"],
                    "size": asset["size"],
                    "sha256": (asset.get("digest") or "").removeprefix("sha256:"),
                }
    raise SystemExit(f"no {backend} asset found in the last 20 llama.cpp releases")


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: Path, size: int, attempts: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    reported = 0.0

    for attempt in range(1, attempts + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= size:
            break

        headers = {"User-Agent": USER_AGENT}
        if have:
            headers["Range"] = f"bytes={have}-"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=90) as resp:
                # A 200 (not 206) means the server ignored Range: start over.
                mode = "ab" if resp.status == 206 and have else "wb"
                if mode == "wb":
                    have = 0
                with dest.open(mode) as fh:
                    while True:
                        chunk = resp.read(1 << 16)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        now = time.time()
                        if now - reported > 15:
                            reported = now
                            elapsed = now - started
                            speed = (have / 1024) / max(elapsed, 1)
                            eta = (size - have) / max(have / max(elapsed, 0.1), 1)
                            print(
                                f"  {have / 1e6:7.1f}/{size / 1e6:.1f} MB "
                                f"({have * 100 // size}%)  {speed:.0f} KB/s  "
                                f"ETA {eta / 60:.0f} min",
                                flush=True,
                            )
        except (urllib.error.URLError, http.client.HTTPException, OSError, TimeoutError) as exc:
            if getattr(exc, "code", None) == 416:
                break
            print(f"  attempt {attempt}: {type(exc).__name__}, resuming", flush=True)
            time.sleep(min(2 * attempt, 15))
            continue

        if dest.stat().st_size >= size:
            break
        print(f"  attempt {attempt}: connection closed early, resuming", flush=True)
        time.sleep(2)
    else:
        raise SystemExit(f"gave up after {attempts} attempts: {dest}")

    print(f"downloaded {dest.name} in {(time.time() - started) / 60:.1f} min", flush=True)


def collect_cuda_libs(target: Path) -> None:
    """Copy locally available CUDA 13 runtime DLLs next to the server binary."""
    found: dict[str, Path] = {}
    for root in CUDA_LIB_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for name in CUDA_LIB_NAMES:
            if name not in found and (root / name).is_file():
                found[name] = root / name

    missing = [n for n in CUDA_LIB_NAMES if n not in found]
    if missing:
        print(
            f"WARNING: no local copy of {', '.join(missing)}; download the "
            f"cudart-llama-bin-win-cuda-13.x-x64.zip release asset into {target}",
            flush=True,
        )

    for name, src in found.items():
        dst = target / name
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            continue
        shutil.copy2(src, dst)
        print(f"  cuda lib: {src} -> {dst.name}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=sorted(ASSET_PATTERNS))
    parser.add_argument("--force", action="store_true", help="re-extract even if present")
    args = parser.parse_args()

    asset = resolve_asset(args.backend)
    print(f"resolved {asset['tag']}/{asset['name']} ({asset['size'] / 1e6:.1f} MB)", flush=True)

    archive = DOWNLOAD_DIR / asset["name"]
    download(asset["url"], archive, asset["size"])

    if asset["sha256"]:
        actual = sha256_of(archive)
        if actual != asset["sha256"]:
            archive.unlink(missing_ok=True)
            raise SystemExit(f"sha256 mismatch: expected {asset['sha256']}, got {actual}")
        print("sha256 verified", flush=True)
    else:
        print("WARNING: release published no digest, skipping verification", flush=True)

    target = RUNTIME_DIR / f"llama-{args.backend}"
    if target.exists():
        if not args.force:
            print(f"{target} already exists, use --force to replace", flush=True)
            return 0
        shutil.rmtree(target)
    target.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(target)
    print(f"extracted to {target}", flush=True)

    if args.backend == "cuda":
        collect_cuda_libs(target)

    server = next(target.rglob("llama-server.exe"), None)
    if server is None:
        raise SystemExit("llama-server.exe not found after extraction")
    print(f"llama-server: {server}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
