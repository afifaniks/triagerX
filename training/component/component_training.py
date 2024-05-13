import os
import sys

sys.path.append('/home/mdafifal.mamun/notebooks/triagerX')

import argparse
import json

import numpy as np
import pandas as pd
import torch
import yaml
from loguru import logger
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from triagerx.dataset.text_processor import TextProcessor
from triagerx.dataset.triage_dataset import TriageDataset
from triagerx.loss.loss_functions import *
from triagerx.model.lbt_p_deberta import LBTPDeberta
from util.epoch_log_manager import EpochLogManager

tqdm.pandas()


parser = argparse.ArgumentParser(description='Training script arguments')
parser.add_argument('--config', type=str, required=True, help='Path to training config file')
parser.add_argument('--seed', type=int, required=True, help='Random seed')
args = parser.parse_args()

logger.debug(f"Loading training configuration from: {args.config}")
with open(args.config, 'r') as stream:
    config = yaml.safe_load(stream)

# Set each field from the YAML config
use_special_tokens = config.get('use_special_tokens')
use_summary = config.get('use_summary')
use_description = config.get('use_description')
dataset_path = config.get('dataset_path')
target_components = config.get('target_components')
base_transformer_model = config.get('base_transformer_model')
unfrozen_layers = config.get('unfrozen_layers')
seed = args.seed
val_size = config.get('val_size')
test_size = config.get('test_size')
dropout = config.get('dropout')
learning_rate = config.get('learning_rate')
epochs = config.get('epochs')
batch_size = config.get('batch_size')
early_stopping_patience = config.get('early_stopping_patience')
topk_indices = config.get('topk_indices')
run_name = config.get('run_name') + f"_seed{seed}"
weights_save_location = os.path.join(config.get('weights_save_location'), f"{run_name}.pt")
test_report_location = os.path.join(config.get('test_report_location'), f"classification_report_{run_name}.json")
device = ("cuda" if torch.cuda.is_available() else "cpu")
wandb_config = {
    "project": config.get('wandb_project', "openj9-component"), 
    "name": run_name, 
    "config": {
        "learning-rate": learning_rate,
        "architecture": config.get('architecture'),
        "unfrozen_layers": unfrozen_layers,
        "dataset": config.get('dataset'),
        "epochs": epochs,
    }
}
log_manager = EpochLogManager(wandb_config)

raw_df = pd.read_csv(dataset_path)

logger.debug(f"Selected compute device: {device}")
logger.debug(f"Weights will be saved in: {weights_save_location}")
logger.debug(f"Classification reports will be saved in: {test_report_location}")
logger.debug(f"Raw dataset size: {len(raw_df)}")

raw_df = raw_df.rename(columns={"assignees": "owner", "issue_body": "description"})
    

def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["labels"].notna()]
    df = df[~df["issue_url"].str.contains("/pull/")]
    
    df["component"] = df["labels"].apply(TextProcessor.component_split)

    df["text"] = df["issue_title"].progress_apply(
            lambda x: "Bug Title: " + str(x),
        ) # type: ignore
    
    if use_special_tokens:
        logger.info("Adding special tokens...")
        df["description"] = df["description"].progress_apply(TextProcessor.clean_text)

    if use_summary:
        logger.info("Adding summary...")
        df["summary"] = df["summary"].progress_apply(TextProcessor.clean_summary)
        df["text"] = df.progress_apply(
            lambda x: x["text"]
            + "\nBug Summary: "
            + str(x["summary"]),
            axis=1
        ) # type: ignore

    if use_description:
        logger.info("Adding description...")
        df["text"] = df.progress_apply(
            lambda x: x["text"]
            + "\nBug Description: "
            + str(x["description"]),
            axis=1,
        ) # type: ignore
    
    min_length = 15
    df = df[df["text"].str.len().gt(min_length)]

    return df

df = prepare_dataframe(raw_df)
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
label2idx = {label: idx for idx, label in enumerate(sorted(list(df_train["component"].unique())))}
df_train["component_id"] = [label2idx[component] for component in df_train["component"].values]
df_test["component_id"] = [label2idx[component] for component in df_test["component"].values]


df_train, df_val = train_test_split(df_train, test_size=val_size, random_state=seed, shuffle=True)

logger.info(f"Final dataset size - Train: {len(df_train)}, Validation: {len(df_val)}, Test: {len(df_test)}")

# Assert each data partition has all the required components
assert set(df_test.component.unique()) == set(df_val.component.unique()) == set(df_train.component.unique())


class_counts = np.bincount(df_train["component_id"])
num_samples = sum(class_counts)
labels = df_train["component_id"].to_list() # corresponding labels of samples

class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[labels[i]] for i in range(int(num_samples))]
sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

model = LBTPDeberta(
    len(df_train.component_id.unique()), 
    unfrozen_layers=unfrozen_layers, 
    dropout=dropout, 
    base_model=base_transformer_model
)

criterion = CombinedLoss()
optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=0.001)
tokenizer = model.tokenizer()

model = model.to(device)
criterion = criterion.to(device)

if use_special_tokens:
    special_tokens = TextProcessor.SPECIAL_TOKENS
    logger.debug("Resizing model embedding for new special tokens...")
    special_tokens_dict = {"additional_special_tokens": list(special_tokens.values())}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.base_model.resize_token_embeddings(len(tokenizer))


train_ds = TriageDataset(df_train, tokenizer, "text", "component_id")
val_ds = TriageDataset(df_val, tokenizer, "text", "component_id")
test_ds = TriageDataset(df_test, tokenizer, "text", "component_id")


train_dataloader = DataLoader(
    dataset=train_ds,
    batch_size=batch_size,
    shuffle=False if sampler else True,
    sampler=sampler,
)
val_dataloader = DataLoader(val_ds, batch_size=batch_size)
test_loader = DataLoader(test_ds, batch_size=batch_size)

total_steps = len(train_dataloader) * epochs
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
best_loss = float("inf")


logger.info("Initiating training...")

patience_counter = 0

for epoch_num in range(epochs):
    total_acc_train = 0
    total_loss_train = 0

    for train_input, train_label in tqdm(train_dataloader, desc="Training Steps"):
        optimizer.zero_grad()

        train_label = train_label.to(device)
        mask = train_input[1]["attention_mask"].squeeze(1).to(device)
        input_id = train_input[1]["input_ids"].squeeze(1).to(device)
        tok_type = train_input[1]["token_type_ids"].squeeze(1).to(device)

        output = model(input_id, mask, tok_type)

        batch_loss = criterion(output, train_label.long())
        total_loss_train += batch_loss.item()

        output = torch.sum(torch.stack(output), 0)
        acc = (output.argmax(dim=1) == train_label).sum().item()
        
        total_acc_train += acc

        batch_loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

    total_acc_val = 0
    total_loss_val = 0
    correct_top_k = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for val_input, val_label in tqdm(val_dataloader, desc="Validation Steps"):
            val_label = val_label.to(device)
            mask = val_input[1]["attention_mask"].squeeze(1).to(device)
            input_id = val_input[1]["input_ids"].squeeze(1).to(device)
            tok_type = val_input[1]["token_type_ids"].squeeze(1).to(device)

            output = model(input_id, mask, tok_type)

            batch_loss = criterion(output, val_label.long())
            total_loss_val += batch_loss.item()

            output = torch.sum(torch.stack(output), 0)
            _, top_k_predictions = output.topk(topk_indices, 1, True, True)

            top_k_predictions = top_k_predictions.t()

            correct_top_k += (
                top_k_predictions.eq(
                    val_label.view(1, -1).expand_as(top_k_predictions)
                )
                .sum()
                .item()
            )

            acc = (output.argmax(dim=1) == val_label).sum().item()

            all_preds.append(output.argmax(dim=1).cpu().numpy())
            all_labels.append(val_label.cpu().numpy())

            total_acc_val += acc

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    precision, recall, f1_score, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro"
    )

    topk = correct_top_k / len(df_val)

    log_manager.log_epoch(
        epoch_num,
        total_acc_train,
        total_acc_val,
        total_loss_train,
        total_loss_val,
        precision,
        recall,
        f1_score,
        df_train,
        df_val,
        (topk_indices, topk)
    )

    val_loss = total_loss_val / len(df_val)

    if val_loss < best_loss:
        patience_counter = 0
        logger.success("Found new best model. Saving weights...")
        torch.save(model.state_dict(), weights_save_location)
        best_loss = val_loss
    else:
        patience_counter += 1
        if patience_counter > early_stopping_patience:
            logger.info("Early stopping...")
            break


log_manager.finish()

# Load best model
model.load_state_dict(torch.load(weights_save_location))


total_acc_val = 0
total_loss_val = 0
correct_top_k = 0
correct_top_k_wo_sim = 0

all_preds = []
all_labels = []
device="cuda"

model = model.cuda()

with torch.no_grad():

    for val_input, val_label in test_loader:
        val_label = val_label.to(device)
        mask = val_input[1]["attention_mask"].squeeze(1).to(device)
        input_id = val_input[1]["input_ids"].squeeze(1).to(device)
        tok_type = val_input[1]["token_type_ids"].squeeze(1).to(device)

        output = model(input_id, mask, tok_type)

        output = torch.sum(torch.stack(output), 0)

        #wo similarity
        _, top_k_wo_sim = output.topk(1, 1, True, True)

        top_k_wo_sim = top_k_wo_sim.t()

        correct_top_k_wo_sim += (
            top_k_wo_sim.eq(
                val_label.view(1, -1).expand_as(top_k_wo_sim)
            )
            .sum()
            .item()
        )

        all_preds.append(output.argmax(dim=1).cpu().numpy())
        all_labels.append(val_label.cpu().numpy())

logger.info(f"Correct Prediction without Similarity: {correct_top_k_wo_sim}, {correct_top_k_wo_sim / len(df_test)}")

all_preds_np = np.concatenate(all_preds)
all_labels_np = np.concatenate(all_labels)

report = classification_report(all_labels_np, all_preds_np, output_dict=True)
report["test_accuracy"] = correct_top_k_wo_sim / len(df_test)
report["run_name"] = run_name
report["model_location"] = weights_save_location

with open(test_report_location, "w") as output_file:
    json.dump(report, output_file)

logger.info(f"Classification Report:\n{report}")
logger.info(f"Classification report saved at: {test_report_location}")