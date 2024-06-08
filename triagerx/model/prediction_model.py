from abc import ABC, abstractmethod


class PredictionModel(ABC):
    @abstractmethod
    def tokenize_text(text):
        raise NotImplementedError()

    @abstractmethod
    def get_label_map():
        raise NotImplementedError()
