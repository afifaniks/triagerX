# %%
import numpy as np
import torch.nn as nn
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler

from triagerx.dataset.processor import DatasetProcessor
from triagerx.model.roberta_fcn import RobertaFCNClassifier
from triagerx.trainer.model_trainer import ModelTrainer
from triagerx.trainer.train_config import TrainConfig

# %% [markdown]
# # Load Data

# %%
dataset_path = "/home/mdafifal.mamun/notebooks/triagerX/notebook/data/deeptriage/classifier_data_20.csv"
NUM_DEVELOPERS = 500
np.random.seed(42)
# %%
import random

# %%
import pandas as pd


def clean_data(df):
    df["text"] = df["text"].str.replace(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " ",
    )
    df["text"] = df["text"].str.replace(" +", " ", regex=True)

    return df


def prepare_dataframe(df: pd.DataFrame, minimum_contribution=300) -> pd.DataFrame:
    # developers = df["owner"].value_counts()
    # filtered_developers = developers.index[developers >= minimum_contribution]
    # df = df[df["owner"].isin(filtered_developers)]

    df["text"] = df.apply(
        lambda x: "Title: "
        + str(x["issue_title"])
        + "\nDescription: "
        + str(x["description"]),
        axis=1,
    )

    min_length = 15
    df = df[df["text"].str.len().gt(min_length)]

    owners = list(set(df["owner"]))
    logger.info(f"Selecting random {NUM_DEVELOPERS} developers...")
    keep_random = np.random.choice(owners, NUM_DEVELOPERS)
    df = df[df["owner"].isin(keep_random)]

    df["owner_id"] = pd.factorize(df["owner"])[0]

    return df


# %%
df = pd.read_csv(dataset_path)
df = prepare_dataframe(df)
df = clean_data(df)

# %%
df["owner"].value_counts().plot(kind="bar")

# %%
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df["owner_id"])
train_df, valid_df = train_test_split(
    train_df, test_size=0.2, stratify=train_df["owner_id"]
)

# %%
assert (
    len(train_df.owner_id.unique())
    == len(train_df.owner_id.unique())
    == len(valid_df.owner_id.unique())
)

# %% [markdown]
# # Training

# %%
model = RobertaFCNClassifier(
    model_name="roberta-large",
    output_size=len(train_df.owner_id.unique()),
    embed_size=1024,
)

# %%
import torch

# %%
class_counts = np.bincount(train_df["owner_id"])
num_samples = sum(class_counts)
labels = train_df["owner_id"].to_list()  # corresponding labels of samples

class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
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

output_file = f"dt_{model_name}_{20}_{sampler_name}"
output_path = f"/home/mdafifal.mamun/notebooks/triagerX/output/{output_file}.pt"

wandb_config = {
    "project": "triagerx",
    "name": f"{output_file}",
    "config": {
        "learning_rate": learning_rate,
        "architecture": "Roberta-FCN",
        "dataset": "deeptriage",
        "epochs": epochs,
    },
}

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, "min", patience=10, factor=0.1, threshold=1e-8)

train_config = TrainConfig(
    optimizer=optimizer,
    criterion=criterion,
    train_dataset=train_df,
    validation_dataset=valid_df,
    learning_rate=learning_rate,
    batch_size=batch_size,
    epochs=epochs,
    output_file=output_path,
    sampler=sampler,
    scheduler=scheduler,
    wandb=wandb_config,
)

logger.info("Starting training...")
trainer = ModelTrainer(train_config)
trainer.train(model=model)

# # %% [markdown]
# # # Evaluation

# # %%
# import torch

# from triagerx.evaluation.evaluator import Evaluator

# # %%
# model = RobertaCNNClassifier(
#     model_name="roberta-large",
#     output_size=len(train_df.owner_id.unique()),
#     embed_size=1024
# )
# model.load_state_dict(torch.load(output_path))

# evaluator = Evaluator()

# # %%
# from torch.utils.data import DataLoader

# from triagerx.dataset.triage_dataset import TriageDataset

# dataset = TriageDataset(test_df, model.tokenizer())

# # %%
# loader = DataLoader(dataset, 15)

# # %%
# device = "cuda"
# all_preds = []
# all_labels = []

# model = model.cuda()

# with torch.no_grad():

#     for val_input, val_label in loader:
#         val_label = val_label.to(device)
#         mask = val_input["attention_mask"].to(device)
#         input_id = val_input["input_ids"].squeeze(1).to(device)

#         output = model(input_id, mask)
#         output = nn.Softmax(dim=1)(output)
#         conf, classes = output.topk(3, dim=1)

#         batch_loss = criterion(output, val_label.long())
#         # total_loss_val += batch_loss.item()

#         acc = (output.argmax(dim=1) == val_label).sum().item()

#         all_preds.append(classes.cpu().numpy())
#         all_labels.append(val_label.cpu().numpy())

#         # total_acc_val += acc

# # %%
# # all_preds[0]

# # %%
# # all_preds = np.concatenate(all_preds)
# # all_labels = np.concatenate(all_labels)

# # %%
# # set(all_labels)

# # %%
# # all_preds[0]

# # %%
# # sum(all_preds == all_labels)

# # %%
# # set(all_labels) - set(all_preds)

# # %%
# # Top 3 Predictions
# evaluator.calculate_top_k_accuray(model, k=3, X_test=test_df, y_test=test_df["owner_id"].to_numpy())

# # %%
# # Top 5 Predictions
# evaluator.calculate_top_k_accuray(model, k=5, X_test=test_df, y_test=test_df["owner_id"].to_numpy())

# %%


# %%
