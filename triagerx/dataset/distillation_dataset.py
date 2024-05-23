import pandas as pd
from torch.utils.data import Dataset


class DistillationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        feature: str,
    ):
        self.tokenizer = tokenizer
        self.texts = [
            self.tokenizer(
                row[feature],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )
            for _, row in df.iterrows()
        ]

    def __len__(self):
        return len(self.texts)

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)

        return batch_texts
