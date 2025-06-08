"""
Example Usage:
python reproduction/lbtp/train_lbtp.py --dataset_path data/deeptriage/google_chrome/classifier_data_20.csv --embedding_model_weights models/distillation/lbtp_gc_base.pt --output_model_weights models/lbtp_dt_gc/lbtp_gc_block9.pt --block 9
"""

import argparse
import sys

sys.path.append("/home/mdafifal.mamun/notebooks/triagerX")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

import wandb
from triagerx.dataset.text_processor import TextProcessor
from triagerx.trainer.model_evaluator import ModelEvaluator

parser = argparse.ArgumentParser(description="Script to run model training")

# Define arguments
parser.add_argument(
    "--dataset_path", type=str, required=True, help="Path to the dataset CSV file"
)
parser.add_argument(
    "--embedding_model_weights",
    type=str,
    required=True,
    help="Directory for the embedding model weights",
)
parser.add_argument("--block", type=int, required=True, default=9, help="Block number")
parser.add_argument(
    "--output_model_weights",
    type=str,
    required=True,
    help="Path for saving the output model weights",
)
parser.add_argument("--run_name", type=str, required=True, help="Run name")
parser.add_argument(
    "--wandb_project", type=str, required=True, help="wandb_project name"
)

# Parse arguments
args = parser.parse_args()

block = args.block
run_name = f"{args.run_name}_block{block}_comp6"
dataset_path = args.dataset_path
embedding_model_weights_dir = args.embedding_model_weights
output_model_weights = args.output_model_weights
test_report_location = (
    f"/home/mdafifal.mamun/notebooks/triagerX/training/reports/{run_name}.json"
)


class TriageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: RobertaTokenizer,
        feature: str = "text",
        target: str = "component_id",
        max_tokens: int = 256,
    ):
        print("Generating torch dataset...")
        self.tokenizer = tokenizer
        self.labels = [label for label in df[target]]
        print("Tokenizing texts...")
        self.texts = [
            self.tokenizer(
                row[feature],
                padding="max_length",
                max_length=max_tokens,
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


class LBTPClassifier(nn.Module):
    def __init__(
        self,
        embedding_model,
        output_size,
        unfrozen_layers=1,
        num_classifiers=3,
        max_tokens=256,
    ) -> None:
        super().__init__()
        self.base_model = embedding_model

        # Freeze embedding layers
        for p in self.base_model.embeddings.parameters():
            p.requires_grad = False

        # Freeze encoder layers till last {unfrozen_layers} layers
        for i in range(0, self.base_model.config.num_hidden_layers - unfrozen_layers):
            for p in self.base_model.encoder.layer[i].parameters():
                p.requires_grad = False

        filter_sizes = [3, 4, 5, 6]
        self._num_filters = 256
        self._max_tokens = max_tokens
        self._num_classifiers = num_classifiers
        self._embed_size = embedding_model.config.hidden_size
        self.unfrozen_layers = unfrozen_layers
        self.conv_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(1, self._num_filters, (K, self._embed_size)),
                            nn.ReLU(),
                            nn.Flatten(),
                            nn.MaxPool1d(self._max_tokens - (K - 1)),
                            nn.Flatten(start_dim=1),
                        )
                        for K in filter_sizes
                    ]
                )
                for _ in range(self._num_classifiers)
            ]
        )

        self.classifier_weights = nn.Parameter(torch.ones(self._num_classifiers))

        self.classifiers = nn.ModuleList(
            [
                nn.Linear(
                    len(filter_sizes) * self._num_filters + self._embed_size,
                    output_size,
                )
                for _ in range(self._num_classifiers)
            ]
        )

        # Dropout is ommitted as it is not mentioned in the LBTP paper
        # self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input_ids = input["input_ids"].squeeze(1).to(device)
        attention_mask = input["attention_mask"].squeeze(1).to(device)

        outputs = []

        base_out = self.base_model(input_ids, attention_mask=attention_mask)
        pooler_out = base_out.pooler_output.squeeze(0)
        hidden_states = base_out.hidden_states[-self._num_classifiers :]

        for i in range(self._num_classifiers):
            batch_size, sequence_length, hidden_size = hidden_states[i].size()
            x = [
                conv(hidden_states[i].view(batch_size, 1, sequence_length, hidden_size))
                for conv in self.conv_blocks[i]
            ]
            x = torch.cat(x, dim=1)
            x = torch.cat([pooler_out, x], dim=1)
            x = self.classifier_weights[i] * self.classifiers[i](x)

            outputs.append(x)

        return outputs


class CombineLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss()

    def forward(self, prediction, labels) -> torch.Tensor:
        loss = 0

        for i in range(len(prediction)):
            loss += self._ce(prediction[i], labels)
            # print(loss)

        return loss


def _count_correct_predictions(top_k_predictions, val_label):
    """Count correct predictions for a given top-k setting."""
    return (
        top_k_predictions.eq(val_label.view(1, -1).expand_as(top_k_predictions))
        .sum()
        .item()
    )


def log_step(
    epoch_num,
    total_acc_train,
    total_acc_val,
    total_loss_train,
    total_loss_val,
    precision,
    recall,
    f1_score,
    train_data,
    validation_data,
    accuracy_top_k,
):
    log_dict = {
        "train_acc": total_acc_train / len(train_data),
        "val_acc": total_acc_val / len(validation_data),
        "train_loss": total_loss_train / len(train_data),
        "val_loss": total_loss_val / len(validation_data),
        "precision": precision,
        "recall": recall,
        "f1-score": f1_score,
    }
    for k_index, k_score in accuracy_top_k.items():
        log_dict[f"top{k_index}_acc"] = k_score

    log = f"Epochs: {epoch_num + 1} | Train Loss: {log_dict['train_loss']: .3f} \
                    | Train Accuracy: {log_dict['train_acc']: .3f} \
                    | Val Loss: {log_dict['val_loss']: .3f} \
                    | Val Accuracy: {log_dict['val_acc']: .3f} \
                    | Top k: {accuracy_top_k} \
                    | Precision: {precision: .3f} \
                    | Recall: {recall: .3f} \
                    | F1-score: {f1_score: .3f}"

    print(log)
    wandb.log(log_dict)


def clean_data(df):
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

    min_length = 15
    df = df[df["text"].str.len().gt(min_length)]

    return df


print("Preparing the dataset...")
df = pd.read_csv(dataset_path)
df = df.rename(columns={"assignees": "owner", "issue_body": "description"})
df = df[df["component"].notna()]
print(set(df["component"].unique()))
# df["component"] = df["labels"].apply(TextProcessor.component_split)
print("Filtering dataset by targetted components...")
target_components = [
    "comp:vm",
    "comp:jit",
    "comp:jvmti",
    "comp:jitserver",
    "comp:jclextensions",
    "comp:test",
    "comp:build",
    "comp:gc",
    "comp:infra",
]
df = df[df["component"].isin(target_components)]

df = clean_data(df)

print(f"Total number of issues: {len(df)}")

num_cv = 10
# sample_threshold=20 # Threshold to filter developers
samples_per_block = len(df) // num_cv

sliced_df = df[: samples_per_block * (block + 1)]

print(f"Samples per block: {samples_per_block}, Selected block: {block}")

# Train and Validation preparation
df_train = sliced_df[: samples_per_block * block]
df_test = sliced_df[samples_per_block * block : samples_per_block * (block + 1)]

# sample_threshold = 20
# developers = df_train["owner"].value_counts()
# filtered_developers = developers.index[developers >= sample_threshold]
# df_train = df_train[df_train["owner"].isin(filtered_developers)]

# train_owners = set(df_train["owner"])
# test_owners = set(df_test["owner"])

# unwanted = list(test_owners - train_owners)

# df_test = df_test[~df_test["owner"].isin(unwanted)]

# print(f"Training data: {len(df_train)}, Validation data: {len(df_test)}")
# print(f"Number of train developers: {len(df_train.owner.unique())}")
# print(f"Number of test developers: {len(df_test.owner.unique())}")

# # Label encode developers

# lbl2idx = {}

# train_owners = sorted(train_owners)

# for idx, dev in enumerate(train_owners):
#     lbl2idx[dev] = idx

# df_train["owner_id"] = df_train["owner"].apply(lambda owner: lbl2idx[owner])
# df_test["owner_id"] = df_test["owner"].apply(lambda owner: lbl2idx[owner])


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

print("Label to ID:", label2idx)
# df_train, df_val = train_test_split(
#     df_train, test_size=val_size, random_state=seed, shuffle=True
# )

print(f"Final dataset size - Train: {len(df_train)}, Test: {len(df_test)}")

# Assert each data partition has all the required components
assert set(df_test.component.unique()) == set(df_train.component.unique())


class_counts = np.bincount(df_train["component_id"])
num_samples = sum(class_counts)
labels = df_train["component_id"].to_list()

print("Load pretrained embedding model")
model_config = RobertaConfig.from_pretrained("roberta-large")
model_config.num_hidden_layers = 3
model_config.output_hidden_states = True
embedding_model = RobertaModel(model_config)
embedding_model.load_state_dict(torch.load(embedding_model_weights_dir))
print("Loaded weights from the saved state.")

tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

model = LBTPClassifier(
    embedding_model, output_size=len(df_train.component.unique()), unfrozen_layers=3
)
learning_rate = 0.00001
epochs = 20
batch_size = 10
topk_indices = [1, 2, 3]

wandb_config = {
    "project": args.wandb_project,
    "name": run_name,
    "config": {
        "learning_rate": learning_rate,
        "dataset": "Deeptriage",
        "epochs": epochs,
    },
}
wandb.init(**wandb_config)

criterion = CombineLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train = TriageDataset(df_train, tokenizer)
val = TriageDataset(df_test, tokenizer)
train_dataloader = DataLoader(
    dataset=train, batch_size=batch_size, shuffle=True, drop_last=True
)
val_dataloader = DataLoader(val, batch_size=batch_size, drop_last=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
best_loss = float("inf")

model = model.to(device)
criterion = criterion.to(device)

for epoch_num in range(epochs):
    total_acc_train = 0
    total_loss_train = 0

    model.train()
    for train_input, train_label in tqdm(train_dataloader, desc="Training Steps"):
        # print(train_input)
        train_label = train_label.to(device)
        # mask = train_input["attention_mask"].squeeze(1).to(device)
        # input_id = train_input["input_ids"].squeeze(1).to(device)

        output = model(train_input)

        batch_loss = criterion(output, train_label.long())
        total_loss_train += batch_loss.item()

        output = torch.sum(torch.stack(output), 0)
        acc = (output.argmax(dim=1) == train_label).sum().item()

        total_acc_train += acc

        model.zero_grad()
        batch_loss.backward()
        optimizer.step()

    total_acc_val = 0
    total_loss_val = 0
    correct_top_k = {k: 0 for k in topk_indices}

    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():

        for val_input, val_label in tqdm(val_dataloader, desc="Validation Steps"):
            val_label = val_label.to(device)
            # input_id = val_input["input_ids"].squeeze(1).to(device)
            # mask = val_input["attention_mask"].squeeze(1).to(device)

            output = model(val_input)

            batch_loss = criterion(output, val_label.long())
            total_loss_val += batch_loss.item()

            output = torch.sum(torch.stack(output), 0)

            _, top_k_predictions = output.topk(max(topk_indices), 1, True, True)
            top_k_predictions = top_k_predictions.t()
            for k in topk_indices:
                correct_top_k[k] += _count_correct_predictions(
                    top_k_predictions[:k], val_label
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

    accuracy_top_k = {
        k: correct_top_k[k] / len(val_dataloader.dataset) for k in topk_indices
    }

    log_step(
        epoch_num,
        total_acc_train,
        total_acc_val,
        total_loss_train,
        total_loss_val,
        precision,
        recall,
        f1_score,
        df_train,
        df_test,
        accuracy_top_k,
    )

    val_loss = total_loss_val / len(df_train)

    if val_loss < best_loss:
        print("Found new best model. Saving weights...")
        torch.save(model.state_dict(), output_model_weights)
        best_loss = val_loss


print("Starting testing...")
model.load_state_dict(torch.load(output_model_weights))

model_evaluator = ModelEvaluator()
model_evaluator.evaluate(
    model=model,
    dataloader=val_dataloader,
    device=device,
    run_name=run_name,
    topk_indices=topk_indices,
    weights_save_location=output_model_weights,
    test_report_location=test_report_location,
)
print("Finished testing.")
