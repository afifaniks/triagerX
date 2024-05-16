import os
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv
from loguru import logger

import wandb

load_dotenv()


class EpochLogManager:
    def __init__(self, wandb_config: Optional[Dict] = None) -> None:
        self._wandb_enabled = False
        if wandb_config:
            wandb.login(key=os.environ["WANDB_API_KEY"])
            logger.debug("Initializing wandb...")
            wandb.init(**wandb_config)
            self._wandb_enabled = True

    def log_epoch(self, epoch_num: int, total_epochs: int, metrics: Dict[str, float]):
        log = f"Epochs: {epoch_num + 1}/{total_epochs}"
        log = (
            log
            + " | "
            + " | ".join(f"{key}: {value}" for key, value in metrics.items())
        )

        logger.info(log)

        if self._wandb_enabled:
            wandb.log(metrics)

    def finish(self):
        if self._wandb_enabled:
            logger.debug("Terminating wandb...")
            wandb.finish()
