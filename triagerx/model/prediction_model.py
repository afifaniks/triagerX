from abc import ABC, abstractmethod

import torch.nn as nn


class PredictionModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @abstractmethod
    def forward(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def tokenize_text(text):
        raise NotImplementedError()

    @abstractmethod
    def get_label_map():
        raise NotImplementedError()
