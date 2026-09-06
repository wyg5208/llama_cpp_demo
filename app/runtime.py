"""Owns the llama-server child process: spawn, wait for readiness, shut down."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import httpx

from .config import Settings
from .models import load_active_model

log = logging.getLogger(__name__)

HEALTH_POLL_INTERVAL = 1.0


class RuntimeUnavailable(RuntimeError):
    pass


class LlamaRuntime:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.process: asyncio.subprocess.Process | None = None
        self.state = "stopped"
        self.detail = ""
        self.props: dict = {}
        self._lock = asyncio.Lock()
        saved = load_active_model(settings.active_model_file)
        self.model_path, self.mmproj_path = saved or (
            settings.model_path,
            # An empty MMPROJ_PATH arrives as WindowsPath('.'), which is truthy.
            settings.mmproj_path if settings.mmproj_path.is_file() else None,
        )

    @property
    def can_switch(self) -> bool:
        return not self.settings.uses_external_server

    @property
    def base_url(self) -> str:
        return self.settings.base_url

    def _set_state(self, value: str) -> None:
        """"switching" spans the whole swap, so only a terminal state may end it.

        Without this, stop() would report "stopped" for up to ten seconds mid-switch
        and the UI would show a red dot with the send button still enabled.
        """
        if self.state != "switching" or value in ("ready", "error", "external"):
            self.state = value

    def build_args(self, exe: Path) -> list[str]:
        s = self.settings
        if not self.model_path.is_file():
            raise RuntimeUnavailable(f"模型文件不存在: {self.model_path}")

        args = [
            str(exe),
            "--model", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(s.llama_port),
            "-ngl", str(s.n_gpu_layers),
            "-c", str(s.n_ctx_for(self.model_path)),
            # Single user, single slot. An explicit --parallel N splits -c across
            # the slots (measured: -np 4 -c 16384 -> n_ctx_slot 4096), which would
            # silently quarter every per-model context.
            "-np", "1",
            # Use the chat template embedded in the GGUF; required for images and tools.
            "--jinja",
            "--no-webui",
        ]
        if self.mmproj_path is not None:
            args += ["--mmproj", str(self.mmproj_path)]
        else:
            log.warning("no mmproj for %s, image input disabled", self.model_path.name)
        if s.server_extra_args.strip():
            args += s.server_extra_args.split()
        return args

    async def start(self) -> None:
        if self.settings.uses_external_server:
            self._set_state("external")
            await self._wait_ready(self.settings.server_startup_timeout)
            return

        exe = self.settings.server_exe
        if exe is None:
            raise RuntimeUnavailable(
                f"未找到 llama-server（backend={self.settings.runtime_backend}）。"
                f"先运行: python scripts/fetch_runtime.py {self.settings.runtime_backend}"
            )

        args = self.build_args(exe)
        log.info("starting llama-server: %s", " ".join(args))
        self.settings.server_log.parent.mkdir(parents=True, exist_ok=True)
        log_file = self.settings.server_log.open("w", encoding="utf-8", errors="replace")

        env = os.environ.copy()
        # Keep the bundled/borrowed CUDA DLLs discoverable next to the binary.
        env["PATH"] = f"{exe.parent}{os.pathsep}{env.get('PATH', '')}"
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

        self._set_state("starting")
        self.process = await asyncio.create_subprocess_exec(
            *args, stdout=log_file, stderr=subprocess.STDOUT, env=env, creationflags=flags
        )
        try:
            await self._wait_ready(self.settings.server_startup_timeout)
        except RuntimeUnavailable:
            self.detail = self._tail_log()
            await self.stop()
            raise

    async def _wait_ready(self, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        last = ""
        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                if self.process is not None and self.process.returncode is not None:
                    raise RuntimeUnavailable(
                        f"llama-server 启动即退出（code={self.process.returncode}）。{self._tail_log()}"
                    )
                try:
                    resp = await client.get(f"{self.base_url}/health")
                    if resp.status_code == 200:
                        self._set_state("ready")
                        self.detail = ""
                        await self._load_props(client)
                        return
                    last = resp.json().get("status", str(resp.status_code))
                except (httpx.HTTPError, ValueError):
                    last = "等待 llama-server 启动"
                self._set_state("loading")
                self.detail = last
                await asyncio.sleep(HEALTH_POLL_INTERVAL)
        raise RuntimeUnavailable(
            f"llama-server 在 {timeout}s 内未就绪，最后状态: {last}\n{self._tail_log()}"
        )

    async def _load_props(self, client: httpx.AsyncClient) -> None:
        try:
            resp = await client.get(f"{self.base_url}/props")
            if resp.status_code == 200:
                self.props = resp.json()
        except (httpx.HTTPError, ValueError) as exc:
            log.warning("could not read /props: %s", exc)

    def _tail_log(self, lines: int = 25) -> str:
        try:
            text = self.settings.server_log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        return "\n".join(text.strip().splitlines()[-lines:])

    async def stop(self) -> None:
        if self.process is None or self.process.returncode is not None:
            self.process = None
            self.props = {}
            self._set_state("stopped")
            return
        self.process.terminate()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.process.kill()
            await self.process.wait()
        self.process = None
        self.props = {}
        self._set_state("stopped")

    async def switch_model(self, model_path: Path, mmproj_path: Path | None) -> dict:
        """Reload llama-server with different weights. The conversation is untouched.

        Rolls back to the previous model when the new one fails to load, and raises
        RuntimeUnavailable either way so the caller can tell the browser what happened.
        """
        async with self._lock:
            if model_path == self.model_path and self.state in ("ready", "external"):
                return self.status()

            previous = (self.model_path, self.mmproj_path)
            self.state = "switching"
            self.detail = f"正在加载 {model_path.name}"
            log.info("switching model: %s -> %s", previous[0].name, model_path.name)

            await self.stop()
            self.model_path, self.mmproj_path = model_path, mmproj_path
            try:
                await self.start()
            except RuntimeUnavailable as exc:
                # Capture this before rolling back: start() truncates the server log.
                failure = str(exc)
                log.warning("model switch failed, rolling back: %s", failure)
                await self.stop()
                self.model_path, self.mmproj_path = previous
                try:
                    await self.start()
                except RuntimeUnavailable as exc2:
                    self.state = "error"
                    self.detail = (
                        f"切换到 {model_path.name} 失败，回滚到 {previous[0].name} 也失败。\n"
                        f"新模型：{failure}\n回滚：{exc2}"
                    )
                    raise RuntimeUnavailable(self.detail) from exc2
                self.detail = f"上一次切换到 {model_path.name} 失败，已回滚"
                raise RuntimeUnavailable(
                    f"{model_path.name} 加载失败，已回滚到 {previous[0].name}。\n{failure}"
                ) from exc
            return self.status()

    @property
    def _vision(self) -> bool:
        modalities = self.props.get("modalities")
        if isinstance(modalities, dict):
            return bool(modalities.get("vision"))
        return self.mmproj_path is not None

    @property
    def supports_tools(self) -> bool:
        caps = self.props.get("chat_template_caps")
        if isinstance(caps, dict) and "supports_tools" in caps:
            return bool(caps["supports_tools"])
        # Fail open: props are empty mid-switch, and an external llama-server may
        # predate chat_template_caps. Killing search over a missing field would
        # disable a feature that almost certainly works.
        return True

    def status(self) -> dict:
        s = self.settings
        defaults = self.props.get("default_generation_settings") or {}
        return {
            "state": self.state,
            "detail": self.detail,
            "backend": s.runtime_backend,
            "base_url": self.base_url,
            "model": self.model_path.name,
            "mmproj": self.mmproj_path.name if self.mmproj_path else None,
            "vision": self._vision,
            "tools": self.supports_tools,
            "can_switch": self.can_switch,
            "search_provider": s.resolved_search_provider(),
            "n_ctx": defaults.get("n_ctx") or s.n_ctx_for(self.model_path),
            "n_gpu_layers": defaults.get("n_gpu_layers", s.n_gpu_layers),
        }
