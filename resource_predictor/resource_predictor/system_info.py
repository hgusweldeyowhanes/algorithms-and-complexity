"""Host / runtime features used as model inputs."""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SystemFeatures:
    cpu_count: int
    os_family: str  # Windows / Linux / Darwin / Other
    python_major_minor: float  # e.g. 3.12
    pointer_bits: int  # 32 or 64
    total_ram_gb: float  # 0 if unknown

    def as_vector(self) -> list[float]:
        os_map = {"Windows": 0.0, "Linux": 1.0, "Darwin": 2.0}
        return [
            float(self.cpu_count),
            os_map.get(self.os_family, 3.0),
            self.python_major_minor,
            float(self.pointer_bits),
            self.total_ram_gb,
        ]


FEATURE_NAMES_SYSTEM = (
    "cpu_count",
    "os_family_code",
    "python_major_minor",
    "pointer_bits",
    "total_ram_gb",
)


def _total_ram_gb() -> float:
    try:
        import psutil  # type: ignore

        return psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass
    # Windows fallback via ctypes
    if sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return stat.ullTotalPhys / (1024**3)
        except Exception:
            return 0.0
    return 0.0


def collect_system_features() -> SystemFeatures:
    family = platform.system() or "Other"
    if family not in {"Windows", "Linux", "Darwin"}:
        family = "Other"
    ver = sys.version_info
    return SystemFeatures(
        cpu_count=os.cpu_count() or 1,
        os_family=family,
        python_major_minor=float(f"{ver.major}.{ver.minor}"),
        pointer_bits=8 * struct_size(),
        total_ram_gb=round(_total_ram_gb(), 2),
    )


def struct_size() -> int:
    return 8 if sys.maxsize > 2**32 else 4


def system_report(feats: SystemFeatures | None = None) -> str:
    feats = feats or collect_system_features()
    d = asdict(feats)
    return "\n".join(f"  {k}: {v}" for k, v in d.items())
