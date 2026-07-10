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

    LLM GPUs are chosen FIRST from the best PIX (PCIe bridge) pair to ensure
    NCCL-compatible tensor parallelism.  The embedding GPU is then picked from
    the remaining GPUs.  If only one GPU is free it is shared by both services.
    """
    all_gpus = {gpu.index: gpu for gpu in get_gpus()}
    gpus = [gpu for gpu in all_gpus.values() if gpu.free_memory_gb >= embedding_memory_gb]
    if not gpus:
        raise GPUUnavailableError(f"No GPU has at least {embedding_memory_gb:.1f} GB free for embeddings")

    # --- discover PIX / NVLink pairs from topology --------------------------------
    pairs: dict[int, set[int]] = {gpu.index: {gpu.index} for gpu in gpus}
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
                if label in ("PIX", "NV") and src in all_gpus and dst in all_gpus:
                    pairs.setdefault(src, set()).add(dst)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    # --- helper: total free MB of the best *count* GPUs in a family ---------------
    def _family_best(gpu_list: list, count: int) -> tuple[float, list[int]]:
        top = sorted(gpu_list, key=lambda g: g.free_memory_gb, reverse=True)[:count]
        return (sum(g.free_memory_gb for g in top), [g.index for g in top])

    # --- pick best LLM pair -------------------------------------------------------
    gpu_by_index = {gpu.index: gpu for gpu in gpus}
    best_llm: list[int] | None = None
    best_llm_mem = -1.0
    seen_families: set[frozenset[int]] = set()

    for gpu in gpus:
        family_ids = frozenset(pairs.get(gpu.index, {gpu.index}))
        if family_ids in seen_families:
            continue
        seen_families.add(family_ids)
        family_gpus = [gpu_by_index[i] for i in family_ids if i in gpu_by_index]
        # filter to GPUs with enough free memory for LLM
        eligible = [g for g in family_gpus if g.free_memory_gb >= llm_memory_gb]
        if len(eligible) >= 2:
            mem, idxs = _family_best(eligible, min(max_llm_gpus, 2))
            if len(idxs) >= 2 and mem > best_llm_mem:
                best_llm_mem = mem
                best_llm = idxs

    if best_llm is not None:
        llm_gpus = best_llm
        # embedding GPU: best free memory among GPUs NOT in the LLM set
        remaining = [g for g in gpus if g.index not in set(llm_gpus)]
        embedding_gpu = max(remaining, key=lambda g: g.free_memory_gb) if remaining else \
                        max(gpus, key=lambda g: g.free_memory_gb)
    else:
        # no full pair available — fall back to single LLM GPU (shared or dedicated)
        llm_candidates = sorted(
            [g for g in gpus if g.free_memory_gb >= llm_memory_gb],
            key=lambda g: g.free_memory_gb, reverse=True,
        )
        embedding_gpu = max(gpus, key=lambda g: g.free_memory_gb)
        if llm_candidates and llm_candidates[0].index != embedding_gpu.index:
            llm_gpus = [llm_candidates[0].index]
        else:
            llm_gpus = [embedding_gpu.index]  # share the embedding GPU

    return embedding_gpu.index, llm_gpus
