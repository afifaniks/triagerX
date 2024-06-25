import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.append("/home/mdafifal.mamun/notebooks/triagerX")

from triagerx.dataset.text_processor import TextProcessor
from triagerx.loss.loss_functions import *
from triagerx.model.module_factory import DatasetFactory, ModelFactory
from triagerx.trainer.model_evaluator import ModelEvaluator
from triagerx.trainer.model_trainer import ModelTrainer
from triagerx.trainer.train_config import TrainConfig
from util.epoch_log_manager import EpochLogManager

tqdm.pandas()


parser = argparse.ArgumentParser(description="Training script arguments")
parser.add_argument(
    "--config", type=str, required=True, help="Path to training config file"
)
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path of the dataset"
)
parser.add_argument("--seed", type=int, required=True, help="Random seed")
args = parser.parse_args()

logger.debug(f"Loading training configuration from: {args.config}")
with open(args.config, "r") as stream:
    config = yaml.safe_load(stream)

dataset_path = args.dataset_path
seed = args.seed

# Set each field from the YAML config
use_special_tokens = config.get("use_special_tokens")
use_summary = config.get("use_summary")
use_description = config.get("use_description")
target_components = config.get("target_components")
base_transformer_models = config.get("base_transformer_models")
unfrozen_layers = config.get("unfrozen_layers")
num_classifiers = config.get("num_classifiers")
max_tokens = config.get("max_tokens")
model_key = config.get("model_key")
val_size = config.get("val_size")
test_size = config.get("test_size")
dropout = config.get("dropout")
learning_rate = config.get("learning_rate")
epochs = config.get("epochs")
batch_size = config.get("batch_size")
early_stopping_patience = config.get("early_stopping_patience")
topk_indices = config.get("topk_indices")
run_name = config.get("run_name") + f"_seed{seed}"
weights_save_location = os.path.join(
    config.get("weights_save_location"), f"{run_name}.pt"
)
test_report_location = os.path.join(
    config.get("test_report_location"), f"classification_report_{run_name}.json"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
wandb_config = {
    "project": config.get("wandb_project", "openj9-component"),
    "name": run_name,
    "config": {
        "learning-rate": learning_rate,
        "architecture": config.get("architecture"),
        "unfrozen_layers": unfrozen_layers,
        "dataset": config.get("dataset"),
        "epochs": epochs,
    },
}

log_manager = EpochLogManager(wandb_config)
torch.manual_seed(seed=seed)

raw_df = pd.read_csv(dataset_path)

logger.debug(f"Seed: {seed}")
logger.debug(f"Selected compute device: {device}")
logger.debug(f"Weights will be saved in: {weights_save_location}")
logger.debug(f"Classification reports will be saved in: {test_report_location}")
logger.debug(f"Raw dataset size: {len(raw_df)}")

raw_df = raw_df.rename(columns={"assignees": "owner", "issue_body": "description"})
df = TextProcessor.prepare_dataframe(
    raw_df,
    use_special_tokens=use_special_tokens,
    use_summary=use_summary,
    use_description=use_description,
    component_training=True,
)

df = df.sort_values(by="issue_number")

num_issues = len(df)
logger.info(f"Total number of issues after processing: {num_issues}")

logger.debug("Filtering dataset by targetted components...")
filtered_df = df[df["component"].isin(target_components)]
df = df.sort_values(by="issue_number")


df_train, df_test = train_test_split(filtered_df, test_size=test_size)
assert set(df_train.component.unique()) == set(df_test.component.unique())
logger.info(f"Train dataset size: {len(df_train)}\nTest dataset size: {len(df_test)}")


# Generate component ids
all_components = sorted(list(df_train["component"].unique()))
label2idx = {}
idx2labels = {}

for idx, comp in enumerate(all_components):
    label2idx[comp] = idx
    idx2labels[idx] = comp

df_train["component_id"] = [
    label2idx[component] for component in df_train["component"].values
]
df_test["component_id"] = [
    label2idx[component] for component in df_test["component"].values
]


df_train, df_val = train_test_split(
    df_train, test_size=val_size, random_state=seed, shuffle=True
)

logger.info(
    f"Final dataset size - Train: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}"
)

# Assert each data partition has all the required components
assert (
    set(df_test.component.unique())
    == set(df_val.component.unique())
    == set(df_train.component.unique())
)


class_counts = np.bincount(df_train["component_id"])
num_samples = sum(class_counts)
labels = df_train["component_id"].to_list()  # corresponding labels of samples

class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[labels[i]] for i in range(int(num_samples))]
sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

model = ModelFactory.get_model(
    model_key=model_key,
    output_size=len(df_train.component_id.unique()),
    unfrozen_layers=unfrozen_layers,
    num_classifiers=num_classifiers,
    base_models=base_transformer_models,
    dropout=dropout,
    max_tokens=max_tokens,
    label_map=idx2labels,
)

criterion = CombinedLoss()

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=0.001)

logger.debug("preparing datasets...")
train_ds = DatasetFactory.get_dataset(
    df_train, model, "text", "component_id", max_length=max_tokens
)
val_ds = DatasetFactory.get_dataset(
    df_val, model, "text", "component_id", max_length=max_tokens
)
test_ds = DatasetFactory.get_dataset(
    df_test, model, "text", "component_id", max_length=max_tokens
)

train_dataloader = DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=False if sampler else True,
    sampler=sampler,
)
val_dataloader = DataLoader(val_ds, batch_size=batch_size)
test_dataloader = DataLoader(test_ds, batch_size=batch_size)

logger.debug("Configuring training parameters...")

total_steps = len(train_dataloader) * epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

train_config = TrainConfig(
    model=model,
    train_dataloader=train_dataloader,
    validation_dataloader=val_dataloader,
    optimizer=optimizer,
    criterion=criterion,
    learning_rate=learning_rate,
    batch_size=batch_size,
    epochs=epochs,
    output_path=weights_save_location,
    device=device,
    topk_indices=topk_indices,
    log_manager=log_manager,
    early_stopping_patience=early_stopping_patience,
    scheduler=scheduler,
)

logger.info("Starting training...")

model_trainer = ModelTrainer(train_config)
model_trainer.train()
log_manager.finish()

logger.info("Finished training.")

# Test
logger.info("Starting testing...")
model.load_state_dict(torch.load(weights_save_location))

model_evaluator = ModelEvaluator()
model_evaluator.evaluate(
    model=model,
    dataloader=test_dataloader,
    device=device,
    run_name=run_name,
    topk_indices=topk_indices,
    weights_save_location=weights_save_location,
    test_report_location=test_report_location,
)
logger.info("Finished testing.")
