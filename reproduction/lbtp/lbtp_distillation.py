"""
Script to distill roberta-large following LBT-P
Example Usage:
```
python reproduction/lbtp/lbtp_distillation.py --dataset_path data/deeptriage/mozilla_core/deep_data.csv --model_weights_path models/distillation/lbtp_mc_base.pt
```
"""

import argparse
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

sys.path.append("/home/mdafifal.mamun/notebooks/triagerX")
from triagerx.dataset.distillation_dataset import DistillationDataset

MODEL_ID = "roberta-large"

teacher_model = RobertaModel.from_pretrained(MODEL_ID, output_hidden_states=True)
student_config = RobertaConfig.from_pretrained(MODEL_ID)
student_config.num_hidden_layers = 3
student_config.output_hidden_states = True


student_model = RobertaModel(student_config)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_ID)

parser = argparse.ArgumentParser(description="Distillation parameters")
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path of the dataset"
)
parser.add_argument(
    "--model_weights_path",
    type=str,
    required=True,
    help="Output path for the model weights",
)
args = parser.parse_args()

open_data = args.dataset_path
model_weights_dir = args.model_weights_path


df = pd.read_csv(open_data)
print(len(df))

df = df.rename(columns={"assignees": "owner", "issue_body": "description"})


def clean_distill_data(df):
    df["text"] = df.apply(
        lambda x: str(x["issue_title"]) + "\n" + str(x["description"]),
        axis=1,
    )
    df["text"] = df["text"].str.replace(
        "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " ",
        regex=True,
    )
    df["text"] = df["text"].str.replace("[^A-Za-z0-9 ]+", " ", regex=True)
    df["text"] = df["text"].str.replace(" +", " ", regex=True)

    return df


df = clean_distill_data(df)
dataset = DistillationDataset(df, tokenizer, "text", max_length=256)
distill_dataloader = DataLoader(dataset, shuffle=True, batch_size=10)


# LR is set to 1e-5 (It is assumed as no reference to LR for distillation training)
optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-5)


num_epochs = 2

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


teacher_model = teacher_model.to(device)
student_model = student_model.to(device)


num_teacher_layers = teacher_model.config.num_hidden_layers
num_student_layers = student_model.config.num_hidden_layers
s = num_teacher_layers // num_student_layers

print(f"Skip layers: {s}")


def pkd_loss(student_reps, teacher_reps, s):
    loss = 0
    for i in range(len(student_reps)):
        student_layer = student_reps[i]
        teacher_layer = teacher_reps[i * s]
        student_layer = F.normalize(student_layer, p=2, dim=-1)
        teacher_layer = F.normalize(teacher_layer, p=2, dim=-1)
        loss += F.mse_loss(student_layer, teacher_layer)

    return loss


for epoch in range(num_epochs):
    student_model.train()
    teacher_model.eval()

    # Wrap the dataloader with tqdm for progress bar
    progress_bar = tqdm(distill_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].squeeze(1).to(student_model.device)
        attention_mask = batch["attention_mask"].squeeze(1).to(student_model.device)

        with torch.no_grad():
            teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
            teacher_hidden_states = teacher_outputs.hidden_states

        student_outputs = student_model(input_ids, attention_mask=attention_mask)
        student_hidden_states = student_outputs.hidden_states
        # break

        loss = pkd_loss(student_hidden_states, teacher_hidden_states, s)
        loss.backward()
        optimizer.step()

        # Update progress bar with loss value
        progress_bar.set_postfix(loss=loss.item())

print("Saving student model...")
torch.save(student_model.state_dict(), model_weights_dir)
print(f"Weights saved to: {model_weights_dir}")
