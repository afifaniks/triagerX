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

# Set each field from the YAML config
dataset_path = args.dataset_path
seed = args.seed

use_description = config.get("use_description")
base_transformer_models = config.get("base_transformer_models")
unfrozen_layers = config.get("unfrozen_layers")
num_classifiers = config.get("num_classifiers")
val_size = config.get("val_size")
test_size = config.get("test_size")
dropout = config.get("dropout")
max_tokens = config.get("max_tokens")
model_key = config.get("model_key")
learning_rate = config.get("learning_rate")
epochs = config.get("epochs")
batch_size = config.get("batch_size")
early_stopping_patience = config.get("early_stopping_patience")
topk_indices = config.get("topk_indices")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(seed=seed)

raw_df = pd.read_csv(dataset_path)

logger.debug(f"Seed: {seed}")
logger.debug(f"Selected compute device: {device}")
logger.debug(f"Raw dataset size: {len(raw_df)}")

raw_df = raw_df.rename(columns={"assignees": "owner", "issue_body": "description"})

df = TextProcessor.prepare_dataframe(
    raw_df,
    use_special_tokens=False,
    use_summary=False,
    use_description=use_description,
    component_training=False,
)

df = df.sort_values(by="issue_number")
df = df[df["owner"].notna()]

num_issues = len(df)
logger.info(f"Total number of issues after processing: {num_issues}")

# Define active user map
vm_users = [
    "pshipton",
    "keithc-ca",
    "gacholio",
    "tajila",
    "babsingh",
    "JasonFengJ9",
    "fengxue-IS",
    "hangshao0",
    "theresa-m",
    "ChengJin01",
    "singh264",
    "thallium",
    "ThanHenderson",
]
jvmti_users = ["gacholio", "tajila", "babsingh", "fengxue-IS"]
jclextensions_users = ["JasonFengJ9", "pshipton", "keithc-ca"]
test_users = ["LongyuZhang", "annaibm", "sophiaxu0424", "KapilPowar", "llxia"]
build_users = ["adambrousseau", "mahdipub"]
gc_users = ["dmitripivkine", "amicic", "kangyining", "LinHu2016"]

# Putting them in dictionaries
components = {
    "comp:vm": vm_users,
    "comp:jvmti": jvmti_users,
    "comp:jclextensions": jclextensions_users,
    "comp:test": test_users,
    "comp:build": build_users,
    "comp:gc": gc_users,
}

expected_users = [
    user.lower() for user_list in components.values() for user in user_list
]
df = df[df["owner"].isin(expected_users)]
logger.info(f"Total issues after developer filtering: {len(df)}")

df = df.sort_values(by="issue_number")

df_train, df_test = train_test_split(df, test_size=test_size, shuffle=False)

sample_threshold = 20
developers = df_train["owner"].value_counts()
filtered_developers = developers.index[developers >= sample_threshold]
df_train = df_train[df_train["owner"].isin(filtered_developers)]

train_owners = set(df_train["owner"])
test_owners = set(df_test["owner"])

unwanted = list(test_owners - train_owners)

df_test = df_test[~df_test["owner"].isin(unwanted)]

logger.info(f"Training data: {len(df_train)}, Validation data: {len(df_test)}")
logger.info(f"Number of developers: {len(df_train.owner.unique())}")

logger.info(f"Train dataset size: {len(df_train)}")
logger.info(f"Test dataset size: {len(df_test)}")


# # Generate label ids
lbl2idx = {}
idx2lbl = {}

train_owners = sorted(train_owners)

for idx, dev in enumerate(train_owners):
    lbl2idx[dev] = idx
    idx2lbl[idx] = dev

df_train["owner_id"] = df_train["owner"].apply(lambda owner: lbl2idx[owner])
df_test["owner_id"] = df_test["owner"].apply(lambda owner: lbl2idx[owner])

df_train.to_csv(f"openj9_train_{len(train_owners)}.csv")
df_test.to_csv(f"openj9_test_{len(train_owners)}.csv")

# assert set(df_train.owner.unique()) == set(df_test.owner.unique())


class_counts = np.bincount(df_train["owner_id"])
num_samples = sum(class_counts)
labels = df_train["owner_id"].to_list()  # corresponding labels of samples

class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[labels[i]] for i in range(int(num_samples))]
sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
output_size = len(df_train.owner_id.unique())

logger.debug("Modeling network...")
model = ModelFactory.get_model(
    model_key=model_key,
    output_size=output_size,
    unfrozen_layers=unfrozen_layers,
    num_classifiers=num_classifiers,
    base_models=base_transformer_models,
    dropout=dropout,
    max_tokens=max_tokens,
    label_map=idx2lbl,
)

criterion = CombinedLoss()

optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=0.001)

logger.debug("preparing datasets...")
train_ds = DatasetFactory.get_dataset(
    df_train, model, "text", "owner_id", max_length=max_tokens
)
val_ds = DatasetFactory.get_dataset(
    df_test, model, "text", "owner_id", max_length=max_tokens
)

train_dataloader = DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=False if sampler else True,
    sampler=sampler,
)
val_dataloader = DataLoader(val_ds, batch_size=batch_size)

logger.debug("Configuring training parameters...")

total_steps = len(train_dataloader) * epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)

run_name = config.get("run_name") + f"_{output_size}devs_seed{seed}"
weights_save_location = os.path.join(
    config.get("weights_save_location"), f"{run_name}.pt"
)
test_report_location = os.path.join(
    config.get("test_report_location"), f"classification_report_{run_name}.json"
)
logger.debug(f"Weights will be saved in: {weights_save_location}")
logger.debug(f"Classification reports will be saved in: {test_report_location}")

wandb_config = {
    "project": config.get("wandb_project", "openj9-developer"),
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
    dataloader=val_dataloader,
    device=device,
    run_name=run_name,
    topk_indices=topk_indices,
    weights_save_location=weights_save_location,
    test_report_location=test_report_location,
)
logger.info("Finished testing.")
