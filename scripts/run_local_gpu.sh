#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi
mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr)
if [[ "${#GPU_ROWS[@]}" -eq 0 ]]; then
  exec .venv/bin/python src/zotero_arxiv_daily/main.py --config-name local
fi
EMBED_GPU="${GPU_ROWS[0]%%,*}"
LLM_GPUS=("$EMBED_GPU")
for row in "${GPU_ROWS[@]:1}"; do
  free_mb="${row##*,}"
  gpu="${row%%,*}"
  if (( free_mb >= 24576 )); then LLM_GPUS+=("$gpu"); fi
  ((${#LLM_GPUS[@]} >= 2)) && break
done
LLM_VISIBLE="$(IFS=,; echo "${LLM_GPUS[*]}")"
export EMBEDDING_DEVICE=cuda:0
export LOCAL_LLM_BASE_URL="${LOCAL_LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
if [[ "${#LLM_GPUS[@]}" -eq 1 ]]; then GPU_MEMORY_UTILIZATION="0.60"; else GPU_MEMORY_UTILIZATION="0.88"; fi
echo "Embedding GPU: $EMBED_GPU; vLLM GPUs: $LLM_VISIBLE"
CUDA_VISIBLE_DEVICES="$LLM_VISIBLE" .venv/bin/vllm serve "${LOCAL_LLM_MODEL:-Qwen/Qwen3-32B}" \
  --tensor-parallel-size "${#LLM_GPUS[@]}" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --host 127.0.0.1 --port 8000 &
VLLM_PID=$!
trap 'kill "$VLLM_PID" 2>/dev/null || true' EXIT
for _ in {1..120}; do
  if curl -fsS "$LOCAL_LLM_BASE_URL/models" >/dev/null 2>&1; then break; fi
  sleep 2
done
if ! curl -fsS "$LOCAL_LLM_BASE_URL/models" >/dev/null 2>&1; then echo "vLLM did not become ready" >&2; exit 1; fi
CUDA_VISIBLE_DEVICES="$EMBED_GPU" .venv/bin/python src/zotero_arxiv_daily/main.py --config-name local
