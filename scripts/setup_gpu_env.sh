#!/usr/bin/env bash
set -euo pipefail
uv venv --clear --python 3.13
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu130 torch torchvision
uv pip install --python .venv/bin/python vllm --torch-backend=auto
.venv/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count())'
echo "GPU environment ready. Run scripts/run_local_gpu.sh after configuring .env."
