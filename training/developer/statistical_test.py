import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

sys.path.append("/home/mdafifal.mamun/notebooks/triagerX")


from triagerx.dataset.text_processor import TextProcessor
from triagerx.loss.loss_functions import *
from triagerx.model.module_factory import DatasetFactory, ModelFactory
from triagerx.trainer.model_trainer import ModelTrainer
from triagerx.trainer.statistical_test import StatisticalEvaluator
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

logger.debug(
    f"Config\n=========================\n{config}\n=========================\n"
)
use_description = config.get("use_description")
model1_transformer_models = config.get("model1_transformer_models")
model2_transformer_models = config.get("model2_transformer_models")
model1_unfrozen_layers = config.get("model1_unfrozen_layers")
model1_num_classifiers = config.get("model1_num_classifiers")
model2_unfrozen_layers = config.get("model2_unfrozen_layers")
model2_num_classifiers = config.get("model2_num_classifiers")
val_size = config.get("val_size")
test_size = config.get("test_size")
model1_dropout = config.get("model1_dropout")
model2_dropout = config.get("model2_dropout")
model1_max_tokens = config.get("model1_max_tokens")
model2_max_tokens = config.get("model2_max_tokens")
model1_key = config.get("model1_key")
model2_key = config.get("model2_key")
model1_path = config.get("model1_path")
model2_path = config.get("model2_path")
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
    is_openj9=True if "openj9" in dataset_path else False,
)

df = df[df["owner"].notna()]

num_issues = len(df)
logger.info(f"Total number of issues after processing: {num_issues}")

if "openj9" in dataset_path:
    print("Sorting issues by issue number...")
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

# assert set(df_train.owner.unique()) == set(df_test.owner.unique())


class_counts = np.bincount(df_train["owner_id"])
num_samples = sum(class_counts)
labels = df_train["owner_id"].to_list()  # corresponding labels of samples

class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[labels[i]] for i in range(int(num_samples))]
sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))
sampler = None

logger.debug("Modeling network...")
model = ModelFactory.get_model(
    model_key=model1_key,
    output_size=len(df_train.owner_id.unique()),
    unfrozen_layers=model1_unfrozen_layers,
    num_classifiers=model1_num_classifiers,
    base_models=model1_transformer_models,
    dropout=model1_dropout,
    max_tokens=model1_max_tokens,
    label_map=idx2lbl,
)

model2 = ModelFactory.get_model(
    model_key=model2_key,
    output_size=len(df_train.owner_id.unique()),
    unfrozen_layers=model2_unfrozen_layers,
    num_classifiers=model2_num_classifiers,
    base_models=model2_transformer_models,
    dropout=model2_dropout,
    max_tokens=model2_max_tokens,
    label_map=idx2lbl,
)

combined_loss1 = False if model1_key == "fcn-transformer" else True
combined_loss2 = False if model2_key == "fcn-transformer" else True

val_ds = DatasetFactory.get_dataset(
    df_test, model, "text", "owner_id", max_length=model1_max_tokens
)
val_ds2 = DatasetFactory.get_dataset(
    df_test, model2, "text", "owner_id", max_length=model2_max_tokens
)

val_dataloader = DataLoader(val_ds, batch_size=batch_size)
val_dataloader2 = DataLoader(val_ds2, batch_size=batch_size)


# Test
logger.info("Starting testing...")
model.load_state_dict(torch.load(model1_path, map_location=device))
model2.load_state_dict(torch.load(model2_path, map_location=device))

m1 = model1_path.split("/")[-1].split(".")[0]
m2 = model2_path.split("/")[-1].split(".")[0]
report_file_name = f"statistical_test_report_{m1}_AND_{m2}.json"
logger.info(f"Report file name: {report_file_name}")

model_evaluator = StatisticalEvaluator()
model_evaluator.evaluate(
    model=model,
    model2=model2,
    dataloader=val_dataloader,
    dataloader2=val_dataloader2,
    device=device,
    topk_indices=topk_indices,
    report_file_name=report_file_name,
    combined_loss1=combined_loss1,
    combined_loss2=combined_loss2,
)
logger.info("Finished testing.")
