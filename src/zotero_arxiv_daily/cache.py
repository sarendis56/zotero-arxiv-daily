"""Paper cache for resuming interrupted local runs.

Retrieved papers are cached by date so that if the pipeline fails during
reranking or TLDR generation (e.g. GPU busy), a later retry can skip
the expensive retrieval step and resume from cache.
"""

import pickle
from datetime import date
from pathlib import Path

from loguru import logger

CACHE_DIR = Path("output/cache")


def _cache_path(for_date: date) -> Path:
    return CACHE_DIR / f"{for_date.isoformat()}.pkl"


def save_papers(papers: list, for_date: date) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(for_date)
    with open(path, "wb") as f:
        pickle.dump(papers, f)
    logger.info(f"Cached {len(papers)} papers to {path}")


def load_papers(for_date: date) -> list | None:
    path = _cache_path(for_date)
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            papers = pickle.load(f)
        logger.info(f"Loaded {len(papers)} papers from cache ({path})")
        return papers
    except Exception as exc:
        logger.warning(f"Failed to load paper cache: {exc}")
        return None
