from typing import Optional, Dict

import pandas as pd
from pydantic import BaseModel
from torch import nn


class TrainConfig(BaseModel):
    model: nn.Module
    optimizer: nn.Module
    criterion: nn.Module
    train_dataset: pd.DataFrame
    validation_dataset: pd.DataFrame
    learning_rate: float
    batch_size: int
    epochs: int
    output_file: str
    scheduler: Optional[nn.Module]
    sampler: Optional[nn.Module]
    wandb: Optional[Dict]
