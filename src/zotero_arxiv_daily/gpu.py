"""GPU discovery and resource planning for the local runner."""

from dataclasses import dataclass
import os
import subprocess
import sys
import time
from pathlib import Path


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


def start_vllm(
    model: str,
    port: int,
    llm_gpus: list[int],
    gpu_memory_utilization: float,
) -> subprocess.Popen:
    """Launch vLLM API server on the given GPUs and return the process handle."""
    vllm_bin = str(Path(sys.executable).parent / "vllm")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in llm_gpus)
    tp_size = len(llm_gpus)
    log_path = Path("output") / "vllm.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(str(log_path), "w")
    return subprocess.Popen(
        [
            vllm_bin, "serve", model,
            "--tensor-parallel-size", str(tp_size),
            "--gpu-memory-utilization", str(gpu_memory_utilization),
            "--host", "127.0.0.1",
            "--port", str(port),
        ],
        env=env,
        stdout=log_file,
        stderr=log_file,
    )


def wait_for_vllm(base_url: str, timeout: int = 600) -> bool:
    """Block until the vLLM server at *base_url* responds to /models."""
    import requests as _requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if _requests.get(f"{base_url}/models", timeout=5).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def plan_gpus(*, embedding_memory_gb: float = 4.0, llm_memory_gb: float = 24.0, max_llm_gpus: int = 2) -> tuple[int, list[int]]:
    """Select one embedding GPU and up to two LLM GPUs.

    LLM GPUs are chosen from the same PIX (PCIe bridge) pair when possible,
    avoiding cross-NUMA NCCL communication failures. If only one GPU is free,
    it is shared by both services.
    """
    gpus = [gpu for gpu in get_gpus() if gpu.free_memory_gb >= embedding_memory_gb]
    if not gpus:
        raise GPUUnavailableError(f"No GPU has at least {embedding_memory_gb:.1f} GB free for embeddings")
    embedding_gpu = max(gpus, key=lambda gpu: gpu.free_memory_gb)

    # Build PIX-adjacent pairs from nvidia-smi topology
    pairs: dict[int, list[int]] = {}
    for g in gpus:
        pairs[g.index] = [g.index]  # every GPU is at least adjacent to itself
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            check=True, capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            if not line.startswith("GPU"):
                continue
            parts = line.split()
            src = int(parts[0].lstrip("GPU"))
            for dst, label in enumerate(parts[1:], start=0):
                if label in ("PIX", "NV"):
                    pairs.setdefault(src, []).append(dst)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass  # fall back to free-memory ordering

    def _best_pair(for_gpus: list, count: int) -> list[int]:
        """Pick the *count* GPUs with the most free memory that share a PIX/NV link."""
        if len(for_gpus) < count:
            return [g.index for g in sorted(for_gpus, key=lambda g: g.free_memory_gb, reverse=True)]
        best = None
        best_mem = -1.0
        for g in for_gpus:
            family = {idx for idx in pairs.get(g.index, [g.index])}
            candidates = [c for c in for_gpus if c.index in family]
            if len(candidates) >= count:
                total = sum(c.free_memory_gb for c in sorted(candidates, key=lambda x: x.free_memory_gb, reverse=True)[:count])
                if total > best_mem:
                    best_mem = total
                    best = [c.index for c in sorted(candidates, key=lambda x: x.free_memory_gb, reverse=True)[:count]]
        return best or [g.index for g in sorted(for_gpus, key=lambda g: g.free_memory_gb, reverse=True)[:count]]

    llm_candidates = [g for g in gpus if g.index != embedding_gpu.index and g.free_memory_gb >= llm_memory_gb]
    llm_gpus = _best_pair(llm_candidates, min(max_llm_gpus, 2)) if llm_candidates else [embedding_gpu.index]
    return embedding_gpu.index, llm_gpus
