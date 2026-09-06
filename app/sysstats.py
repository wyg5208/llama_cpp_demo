"""Machine telemetry for the topbar meter: CPU, RAM, GPU, VRAM, GPU temperature.

Plain ctypes against Win32 and NVML — no psutil, no pynvml, and no nvidia-smi
subprocess. A full poll measures ~1 ms in-process; the same numbers through
nvidia-smi cost 51 ms plus a process spawn every time.

CPU temperature is deliberately absent. Reading it on Windows needs a third-party
kernel driver with admin rights, and the only ACPI thermal zone this machine
exposes is a constant: saturating all 32 cores for 9 s did not move it off 74 °C.
GPU temperature comes from NVML and is real.
"""

from __future__ import annotations

import ctypes
import logging

log = logging.getLogger(__name__)

GIB = 1024**3
NVML_SENSOR_GPU = 0

# DWORD / BOOL spelled out rather than via ctypes.wintypes, so importing this
# module cannot fail on a platform that has no wintypes; only the WinDLL loads
# below are Windows-specific, and those are guarded.
DWORD = ctypes.c_uint32
BOOL = ctypes.c_int


class _FILETIME(ctypes.Structure):
    _fields_ = [("lo", DWORD), ("hi", DWORD)]


class _MEMORYSTATUSEX(ctypes.Structure):
    _fields_ = [
        ("dwLength", DWORD),
        ("dwMemoryLoad", DWORD),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _NVML_UTILIZATION(ctypes.Structure):
    # Two uints, not one: NVML writes both, so a bare c_uint here would let it
    # write four bytes past the buffer.
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


class _NVML_MEMORY(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def _load(name: str, prototypes: dict) -> ctypes.WinDLL | None:
    """Load a system DLL and declare its signatures, or return None.

    Signatures are declared rather than left to ctypes' default int marshalling,
    which truncates pointer-sized arguments to 32 bits on x64.
    """
    try:
        lib = ctypes.WinDLL(name)
    except (AttributeError, OSError):
        # AttributeError: not Windows. OSError: DLL absent, e.g. no NVIDIA driver.
        return None
    for fname, (argtypes, restype) in prototypes.items():
        fn = getattr(lib, fname)
        fn.argtypes = argtypes
        fn.restype = restype
    return lib


_KERNEL_PROTOTYPES = {
    "GetSystemTimes": ([ctypes.POINTER(_FILETIME)] * 3, BOOL),
    "GlobalMemoryStatusEx": ([ctypes.POINTER(_MEMORYSTATUSEX)], BOOL),
}

_NVML_PROTOTYPES = {
    "nvmlInit_v2": ([], ctypes.c_uint),
    "nvmlDeviceGetHandleByIndex_v2": ([ctypes.c_uint, ctypes.POINTER(ctypes.c_void_p)], ctypes.c_uint),
    "nvmlDeviceGetUtilizationRates": ([ctypes.c_void_p, ctypes.POINTER(_NVML_UTILIZATION)], ctypes.c_uint),
    "nvmlDeviceGetMemoryInfo": ([ctypes.c_void_p, ctypes.POINTER(_NVML_MEMORY)], ctypes.c_uint),
    "nvmlDeviceGetTemperature": ([ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_uint)], ctypes.c_uint),
    "nvmlDeviceGetPowerUsage": ([ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint)], ctypes.c_uint),
}


class SystemStats:
    """Read once per poll. Constructed a single time, at app startup."""

    def __init__(self) -> None:
        self._kernel = _load("kernel32", _KERNEL_PROTOTYPES)
        # CPU load is a delta between two readings, so take the baseline now;
        # otherwise the first poll after startup has nothing to compare against.
        self._prev: tuple[int, int, int] | None = None
        if self._kernel is not None:
            self._cpu_percent()
        self._nvml = self._open_nvml()
        self._gpu: ctypes.c_void_p | None = None
        if self._nvml is not None:
            handle = ctypes.c_void_p()
            if self._nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) == 0:
                self._gpu = handle
            else:
                self._nvml = None
        if self._nvml is None:
            log.info("GPU telemetry unavailable: nvml.dll not usable")

    def _open_nvml(self) -> ctypes.WinDLL | None:
        lib = _load("nvml.dll", _NVML_PROTOTYPES)
        if lib is None or lib.nvmlInit_v2() != 0:
            return None
        return lib

    def _cpu_percent(self) -> float | None:
        idle, kernel, user = _FILETIME(), _FILETIME(), _FILETIME()
        if not self._kernel.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        ):
            return None
        # Windows folds idle into kernel time, so busy subtracts it back out.
        now = (_ticks(kernel), _ticks(user), _ticks(idle))
        prev, self._prev = self._prev, now
        if prev is None:
            return None
        total = (now[0] - prev[0]) + (now[1] - prev[1])
        if not total:
            return None
        busy = total - (now[2] - prev[2])
        return round(100 * busy / total, 1)

    def _ram(self) -> tuple[float | None, float | None]:
        status = _MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
        if not self._kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None, None
        return (
            round((status.ullTotalPhys - status.ullAvailPhys) / GIB, 1),
            round(status.ullTotalPhys / GIB, 1),
        )

    def _gpu_read(self) -> dict:
        out = {"gpu_percent": None, "vram_used_gb": None, "vram_total_gb": None,
               "gpu_temp_c": None, "gpu_watts": None}
        if self._gpu is None:
            return out
        util, mem = _NVML_UTILIZATION(), _NVML_MEMORY()
        temp, watts = ctypes.c_uint(), ctypes.c_uint()
        n = self._nvml
        # A nonzero return means that one reading is stale, not that the GPU has
        # gone away, so each field is filled independently.
        if n.nvmlDeviceGetUtilizationRates(self._gpu, ctypes.byref(util)) == 0:
            out["gpu_percent"] = util.gpu
        if n.nvmlDeviceGetMemoryInfo(self._gpu, ctypes.byref(mem)) == 0:
            out["vram_used_gb"] = round(mem.used / GIB, 2)
            out["vram_total_gb"] = round(mem.total / GIB, 2)
        if n.nvmlDeviceGetTemperature(self._gpu, NVML_SENSOR_GPU, ctypes.byref(temp)) == 0:
            out["gpu_temp_c"] = temp.value
        if n.nvmlDeviceGetPowerUsage(self._gpu, ctypes.byref(watts)) == 0:
            out["gpu_watts"] = round(watts.value / 1000, 1)
        return out

    def read(self) -> dict:
        out: dict = {
            "cpu_percent": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
        }
        if self._kernel is not None:
            out["cpu_percent"] = self._cpu_percent()
            out["ram_used_gb"], out["ram_total_gb"] = self._ram()
        out["gpu_available"] = self._gpu is not None
        out.update(self._gpu_read())
        return out


def _ticks(ft: _FILETIME) -> int:
    return (ft.hi << 32) | ft.lo
