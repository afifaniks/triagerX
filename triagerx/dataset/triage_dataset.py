import pandas as pd
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

from transformers import PreTrainedTokenizer


class TriageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        feature: str,
        target: str,
    ):
        logger.debug("Generating torch dataset...")
        logger.debug(f"Dataset feature column: {feature}, target column: {target}")
        self.tokenizer = tokenizer
        self.labels = [label for label in df[target]]
        logger.debug("Tokenizing texts...")
        self.texts = [
            (row[feature], self.tokenizer(
                row[feature],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            ))
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
