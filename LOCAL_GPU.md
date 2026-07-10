# Local GPU runner

This project is configured for the local machine's A100 GPUs. The recommended
embedding model is `BAAI/bge-m3`, with a default batch size of 512. The local
LLM defaults to `Qwen/Qwen3-32B` served by vLLM.

```bash
./scripts/setup_gpu_env.sh
cp .env.example .env  # or create .env manually
./scripts/run_local_gpu.sh
```

Required `.env` values are `ZOTERO_ID`, `ZOTERO_KEY`, `SENDER`,
`RECEIVER`, and `SENDER_PASSWORD`. Optional values include
`LOCAL_LLM_MODEL`, `EMBEDDING_BATCH_SIZE`, `ARXIV_CATEGORIES`, and
`SMTP_SERVER`.

The launcher examines free GPU memory. It reserves one GPU for embeddings and
uses up to two other GPUs for vLLM. If only one GPU is available, vLLM is
started with lower memory utilization and the GPU is shared. If no GPU has
enough free memory for embeddings, the application sends a pause notification
email and stops before contacting Zotero or arXiv.

The recommender uses a 30-day short-term profile weighted at 70%, a recency-
decayed long-term profile weighted at 30%, and a small maximum-similarity term
to preserve strong matches to individual research threads.
