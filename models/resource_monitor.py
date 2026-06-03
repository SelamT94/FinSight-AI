"""System resource snapshots for benchmarking LLM runs."""

from __future__ import annotations

import time
from typing import Any, Callable

import psutil


class ResourceMonitor:
    """Capture RAM/CPU (and optional GPU) usage around a function call."""

    def snapshot(self) -> dict[str, Any]:
        vm = psutil.virtual_memory()
        ram_used_gb = vm.used / (1024**3)
        ram_total_gb = vm.total / (1024**3)
        ram_percent = float(vm.percent)
        cpu_percent = float(psutil.cpu_percent(interval=0.1))

        out: dict[str, Any] = {
            "ram_used_gb": round(ram_used_gb, 4),
            "ram_total_gb": round(ram_total_gb, 4),
            "ram_percent": ram_percent,
            "cpu_percent": cpu_percent,
            "gpu_used_mb": None,
            "gpu_total_mb": None,
        }

        try:
            import GPUtil  # noqa: PLC0415

            gpus = GPUtil.getGPUs()
            if gpus:
                g = gpus[0]
                out["gpu_used_mb"] = float(g.memoryUsed)
                out["gpu_total_mb"] = float(g.memoryTotal)
        except Exception:
            pass

        return out

    def measure(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, dict[str, Any]]:
        """Run ``func`` and report duration plus peak RAM and average CPU across snapshots."""
        s0 = self.snapshot()
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        s1 = self.snapshot()

        duration_sec = t1 - t0
        peak_ram_gb = max(s0["ram_used_gb"], s1["ram_used_gb"])
        cpu_avg_percent = (float(s0["cpu_percent"]) + float(s1["cpu_percent"])) / 2.0

        delta = {
            "duration_sec": float(duration_sec),
            "peak_ram_gb": float(peak_ram_gb),
            "cpu_avg_percent": float(cpu_avg_percent),
            "snapshot_before": s0,
            "snapshot_after": s1,
        }
        return result, delta
