from typing import Any, Optional

from pydantic import BaseModel
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from util.epoch_log_manager import EpochLogManager


class TrainConfig(BaseModel):
    model: nn.Module
    train_dataloader: DataLoader
    validation_dataloader: DataLoader
    optimizer: Optimizer
    criterion: nn.Module
    learning_rate: float
    batch_size: int
    epochs: int
    output_path: str
    device: str
    topk_indices: int
    log_manager: EpochLogManager
    early_stopping_patience: Optional[int] = None
    scheduler: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True
