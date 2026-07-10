"""GPU discovery and resource planning for the local runner."""

from dataclasses import dataclass
import subprocess


class GPUUnavailableError(RuntimeError):
    """Raised when local GPU resources are not sufficient to run safely."""


@dataclass(frozen=True)
class GPUInfo:
    index: int
    free_memory_gb: float
    total_memory_gb: float


def get_gpus() -> list[GPUInfo]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise GPUUnavailableError("nvidia-smi is unavailable or failed") from exc
    gpus = []
    for line in result.stdout.splitlines():
        index, free_mb, total_mb = (int(value.strip()) for value in line.split(","))
        gpus.append(GPUInfo(index, free_mb / 1024, total_mb / 1024))
    if not gpus:
        raise GPUUnavailableError("No NVIDIA GPUs were detected")
    return gpus


def plan_gpus(*, embedding_memory_gb: float = 4.0, llm_memory_gb: float = 24.0, max_llm_gpus: int = 2) -> tuple[int, list[int]]:
    """Select one embedding GPU and up to two LLM GPUs.

    If only one GPU is free, it is shared by both services; the vLLM launcher
    uses conservative memory utilization in that mode.
    """
    gpus = [gpu for gpu in get_gpus() if gpu.free_memory_gb >= embedding_memory_gb]
    if not gpus:
        raise GPUUnavailableError(f"No GPU has at least {embedding_memory_gb:.1f} GB free for embeddings")
    embedding_gpu = max(gpus, key=lambda gpu: gpu.free_memory_gb)
    llm_candidates = [gpu for gpu in gpus if gpu.index != embedding_gpu.index and gpu.free_memory_gb >= llm_memory_gb]
    llm_candidates.sort(key=lambda gpu: gpu.free_memory_gb, reverse=True)
    llm_gpus = [gpu.index for gpu in llm_candidates[:max_llm_gpus]] or [embedding_gpu.index]
    return embedding_gpu.index, llm_gpus
