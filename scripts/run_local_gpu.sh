#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
if [[ -f .env ]]; then set -a; source .env; set +a; fi

# GPU detection is now handled by the Python gpu module at runtime.
# vLLM is started lazily by the executor right before TLDR generation,
# so GPU memory is free during the (lengthy) retrieval + reranking phase.
exec .venv/bin/python src/zotero_arxiv_daily/main.py --config-name local
