#!/usr/bin/env bash
set -euo pipefail
uv venv --clear --python 3.13
uv sync --group gpu
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count())'
.venv/bin/python -c 'from vllm import LLM; print("vLLM OK")'
echo "GPU environment ready. Run scripts/run_local_gpu.sh after configuring .env."
