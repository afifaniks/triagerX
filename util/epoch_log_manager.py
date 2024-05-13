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
    
    def log_epoch(
            self,
            epoch_num,
            total_acc_train,
            total_acc_val,
            total_loss_train,
            total_loss_val,
            precision,
            recall,
            f1_score,
            train_data,
            validation_data,
            topk: Tuple[int, float],
        ):
            log = f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                        | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                        | Val Loss: {total_loss_val / len(validation_data): .3f} \
                        | Val Accuracy: {total_acc_val / len(validation_data): .3f} \
                        | Top {topk[0]}: {topk[1]} \
                        | Precision: {precision: .3f} \
                        | Recall: {recall: .3f} \
                        | F1-score: {f1_score: .3f}"

            logger.info(log)
            
            if self._wandb_enabled:
                 wandb.log({
                    "train_acc": total_acc_train / len(train_data), 
                    "train_loss": total_loss_train / len(train_data),
                    "val_acc": total_acc_val / len(validation_data),
                    "val_loss": total_loss_val / len(validation_data),
                    f"top{topk[0]}_acc": topk[1],
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1_score
                })

    def finish(self):
        logger.debug("Terminating wandb...")
        wandb.finish()
