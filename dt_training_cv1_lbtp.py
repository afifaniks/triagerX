# %%
import numpy as np
# %%
import pandas as pd
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler

from triagerx.dataset.processor import DatasetProcessor
from triagerx.model.lbt_p import LBTPClassifier
from triagerx.model.roberta_cnn import RobertaCNNClassifier
from triagerx.model.roberta_fcn import RobertaFCNClassifier
from triagerx.trainer.model_trainer import ModelTrainer
from triagerx.trainer.train_config import TrainConfig

# %% [markdown]
# # Load Data


# %%
dataset_path = "/home/mdafifal.mamun/notebooks/triagerX/notebook/data/deeptriage/gc_20.json"

# %%
df = pd.read_json(dataset_path)

# %%
len(df)

# %%
df = df[df["owner"].notna()]

# %%
df.head()

# %%
# df = df.sort_values(by="reported_time")

# %%
df.head()

# %%
def clean_data(df):
    df['text'] = df['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')
    df["text"] = df['text'].str.replace(" +", " ", regex=True)

    return df
    
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df.apply(
            lambda x: "Title: "
            + str(x["issue_title"])
            + "\nDescription: "
            + str(x["description"]),
            axis=1,
        )
    
    min_length = 15
    df = df[df["text"].str.len().gt(min_length)]

    # df["owner_id"] = pd.factorize(df["assignees"])[0]

    return df

# %%
df = prepare_dataframe(df)
df = clean_data(df)

# %%
num_issues = len(df)

print(f"Total number of issues: {num_issues}")

# %%
num_cv = 10
samples_per_block = len(df) // num_cv + 1
print(f"Samples per block: {samples_per_block}")
block = 1
# %%
X_df = df[:samples_per_block*block]
y_df = df[samples_per_block*block : samples_per_block * (block+1)]

# %%
train_owners = set(X_df["owner"])
test_owners = set(y_df["owner"])

# %%
unwanted = list(test_owners - train_owners)

# %%
y_df = y_df[~y_df["owner"].isin(unwanted)]

# %%
print(f"Training data: {len(X_df)}, Validation data: {len(y_df)}")

# %%
lbl2idx = {}

# %%
for idx, dev in enumerate(train_owners):
    lbl2idx[dev] = idx

# %%
X_df["owner_id"] = X_df["owner"].apply(lambda owner: lbl2idx[owner])
y_df["owner_id"] = y_df["owner"].apply(lambda owner: lbl2idx[owner])

# %%
import torch


class CombineLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss()

    def forward(
        self,
        prediction,
        labels
    ) -> torch.Tensor:
        loss = 0

        for i in range(len(prediction)):
            loss += self._ce(prediction[i], labels)
            # print(loss)

        return loss

# %%
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.tokenizer = tokenizer
        self.labels = [label for label in df['owner_id']]
        self.texts = [self.tokenizer(text,
                               padding='max_length', max_length = 512, truncation=True,
                                return_tensors="pt") for text in df['text']]

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

# %%
model = LBTPClassifier(
    output_size=len(X_df.owner_id.unique())
)

# %%
import torch

# %%
#Let there be 9 samples and 1 sample in class 0 and 1 respectively
class_counts = np.bincount(X_df["owner_id"])
num_samples = sum(class_counts)
labels = X_df["owner_id"].to_list() #corresponding labels of samples

class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[labels[i]] for i in range(int(num_samples))]
sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

# %%
learning_rate = 1e-5
epochs = 50
batch_size = 15

# Create sampler
# counts = np.bincount(train_df["owner_id"])
# labels_weights = 1. / counts
# weights = labels_weights[train_df["owner_id"]]
# sampler = WeightedRandomSampler(weights, len(weights))

sampler_name = sampler.__class__.__name__ if sampler else "None"
model_name = model.__class__.__name__

output_file = f"dt_lbtp_cv{block}_{model_name}_20_{sampler_name}"
output_path = f"/home/mdafifal.mamun/notebooks/triagerX/output/{output_file}.pt"

wandb_config = {
        "project": "triagerx_dt_cv",
        "name": f"run_{output_file}",
        "config": {
        "learning_rate": learning_rate,
        "architecture": "Roberta-FCN",
        "dataset": "deeptriage",
        "epochs": epochs,
    }
}

criterion = CombineLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.1, threshold=1e-8)

train_config = TrainConfig(
    optimizer=optimizer,
    criterion=criterion,
    train_dataset=X_df,
    validation_dataset=y_df,
    learning_rate=learning_rate,
    batch_size=batch_size,
    epochs=epochs,
    output_file=output_path,
    sampler=sampler,
    scheduler=scheduler,
    wandb=wandb_config
)

trainer = ModelTrainer(train_config)
trainer.train(model=model)
