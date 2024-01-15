from typing import Dict, Optional

import pandas as pd
from pydantic import BaseModel
from torch import nn
from torch.optim import Optimizer


class TrainConfig(BaseModel):
    optimizer: Optimizer
    criterion: nn.Module
    train_dataset: pd.DataFrame
    validation_dataset: pd.DataFrame
    learning_rate: float
    batch_size: int
    epochs: int
    output_file: str
    scheduler: Optional[nn.Module] = None
    sampler: Optional[nn.Module] = None
    wandb: Optional[Dict] = None

    class Config:
        arbitrary_types_allowed = True
