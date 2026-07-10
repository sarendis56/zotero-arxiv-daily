import os
import sys
import logging
from omegaconf import DictConfig
import hydra
from loguru import logger
import dotenv
from zotero_arxiv_daily.executor import Executor
from zotero_arxiv_daily.gpu import GPUUnavailableError, plan_gpus
from zotero_arxiv_daily.notifications import send_gpu_unavailable_notification
os.environ["TOKENIZERS_PARALLELISM"] = "false"
dotenv.load_dotenv()

@hydra.main(version_base=None, config_path="../../config", config_name="default")
def main(config:DictConfig):
    # Configure loguru log level based on config
    log_level = "DEBUG" if config.executor.debug else "INFO"
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    for logger_name in logging.root.manager.loggerDict:
        if "zotero_arxiv_daily" in logger_name:
            continue
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    if config.executor.debug:
        logger.info("Debug mode is enabled")
    
    if config.get("runtime", {}).get("gpu") is not None:
        try:
            embedding_gpu, llm_gpus = plan_gpus(
                embedding_memory_gb=config.runtime.gpu.embedding_memory_gb,
                llm_memory_gb=config.runtime.gpu.llm_memory_gb,
                max_llm_gpus=config.runtime.gpu.max_llm_gpus,
            )
            logger.info(f"GPU preflight passed: embedding GPU {embedding_gpu}; LLM GPUs {llm_gpus}")
        except GPUUnavailableError as exc:
            logger.error(str(exc))
            send_gpu_unavailable_notification(config, str(exc))
            return
    executor = Executor(config)
    executor.run()

if __name__ == '__main__':
    main()