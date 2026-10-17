"""The machine a latency number was measured on (latency is meaningless without it)."""

import os
import platform


def cpu_name() -> str:
    if platform.system() == "Windows":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def hardware_info() -> dict:
    import torch  # slow import, only when measuring

    return {
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "torch": torch.__version__,
        "gpu": False,
        "os": platform.platform(),
        "python": platform.python_version(),
    }


def hardware_line(info: dict) -> str:
    return (
        f"{info['cpu']}, {info['logical_cpus']} logical CPUs, torch {info['torch']} on CPU with "
        f"{info['torch_threads']} threads, {info['os']}"
    )
