import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from triagerx.model.prediction_model import PredictionModel


class TriageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model: PredictionModel,
        feature: str,
        target: str,
        max_length: int = 512,
    ):
        logger.debug("Generating torch dataset...")
        logger.debug(f"Dataset feature column: {feature}, target column: {target}")
        self.tokenizer = model.tokenizer()
        self.labels = [label for label in df[target]]
        logger.debug("Tokenizing texts...")
        self.texts = [
            self.tokenizer(
                row[feature],
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            for _, row in df.iterrows()
        ]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
