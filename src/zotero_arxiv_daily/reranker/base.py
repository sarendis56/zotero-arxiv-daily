from abc import ABC, abstractmethod
from omegaconf import DictConfig
from ..protocol import Paper, CorpusPaper
import numpy as np
from typing import Type
from datetime import datetime, timedelta
class BaseReranker(ABC):
    def __init__(self, config:DictConfig):
        self.config = config

    def rerank(self, candidates:list[Paper], corpus:list[CorpusPaper]) -> list[Paper]:
        corpus = sorted(corpus,key=lambda x: x.added_date,reverse=True)
        sim = self.get_similarity_score([c.abstract for c in candidates], [c.abstract for c in corpus])
        assert sim.shape == (len(candidates), len(corpus))
        now = datetime.now()
        executor_config = getattr(self.config, "executor", None)
        recent_days = executor_config.get("recent_interest_days", 30) if executor_config is not None else 30
        recent_weight = executor_config.get("recent_interest_weight", 0.70) if executor_config is not None else 0.70
        long_term_weight = 1.0 - recent_weight
        recent_mask = np.array([now - paper.added_date <= timedelta(days=recent_days) for paper in corpus]) if now is not None else np.zeros(len(corpus), dtype=bool)
        long_decay = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
        long_decay = long_decay / long_decay.sum()
        if recent_mask.any():
            recent_decay = np.exp(-np.arange(len(corpus)) / max(recent_days / 7, 1)) * recent_mask
            recent_decay = recent_decay / recent_decay.sum()
            recent_scores = (sim * recent_decay).sum(axis=1)
        else:
            recent_scores = (sim * long_decay).sum(axis=1)
        long_scores = (sim * long_decay).sum(axis=1)
        max_scores = sim.max(axis=1)
        max_weight = executor_config.get("max_similarity_weight", 0.10) if executor_config is not None else 0.10
        scores = ((1.0 - max_weight) * (recent_weight * recent_scores + long_term_weight * long_scores) + max_weight * max_scores) * 10
        for s,c in zip(scores,candidates):
            c.score = s
        candidates = sorted(candidates,key=lambda x: x.score,reverse=True)
        return candidates
    
    @abstractmethod
    def get_similarity_score(self, s1:list[str], s2:list[str]) -> np.ndarray:
        raise NotImplementedError

registered_rerankers = {}

def register_reranker(name:str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls
    return decorator

def get_reranker_cls(name:str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]