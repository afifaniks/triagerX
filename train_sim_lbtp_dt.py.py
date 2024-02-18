# %%
import numpy as np
import pandas as pd
import torch
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

# %%
dataset_path = "/home/mdafifal.mamun/notebooks/triagerX/notebook/data/deeptriage/gc_20.json"

# %%

dataset_path = "/home/mdafifal.mamun/notebooks/triagerX/notebook/data/deeptriage/gc_20.json"

df = pd.read_json(dataset_path)
df = df[df["owner"].notna()]

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

df = prepare_dataframe(df)
df = clean_data(df)

num_issues = len(df)

print(f"Total number of issues: {num_issues}")

# %%
num_cv = 10
sample_threshold=20
samples_per_block = len(df) // num_cv + 1
print(f"Samples per block: {samples_per_block}")

block = 1
X_df = df[:samples_per_block*block]
y_df = df[samples_per_block*block : samples_per_block * (block+1)]


developers = X_df["owner"].value_counts()
filtered_developers = developers.index[developers >= sample_threshold]
X_df = X_df[X_df["owner"].isin(filtered_developers)]

train_owners = set(X_df["owner"])
test_owners = set(y_df["owner"])

unwanted = list(test_owners - train_owners)

y_df = y_df[~y_df["owner"].isin(unwanted)]

print(f"Training data: {len(X_df)}, Validation data: {len(y_df)}")

# %%
lbl2idx = {}

train_owners = sorted(train_owners)

for idx, dev in enumerate(train_owners):
    lbl2idx[dev] = idx

# %%
X_df["owner_id"] = X_df["owner"].apply(lambda owner: lbl2idx[owner])
y_df["owner_id"] = y_df["owner"].apply(lambda owner: lbl2idx[owner])

# %%
class CombineLoss(nn.Module):
    def __init__(self, class_weights) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss(weight=class_weights)

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
model = LBTPClassifier(
    output_size=len(X_df.owner_id.unique())
)

# %%
class_counts = np.bincount(X_df["owner_id"])
num_samples = sum(class_counts)
labels = X_df["owner_id"].to_list() #corresponding labels of samples

class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
weights = [class_weights[labels[i]] for i in range(int(num_samples))]
sampler = WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

# %%
from sklearn.utils.class_weight import compute_class_weight

# %%
class_weights = compute_class_weight('balanced', classes=X_df["owner_id"].unique(), y=X_df["owner_id"].to_numpy())

# Convert class weights to a tensor
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# %%
learning_rate = 1e-5
epochs = 15
batch_size = 15

# %%
sampler_name = sampler.__class__.__name__ if sampler else "None"
model_name = model.__class__.__name__

output_file = f"dt_lbtp_cv{block}_weighted_ce_{model_name}_20_{sampler_name}"
output_path = f"/home/mdafifal.mamun/notebooks/triagerX/output/{output_file}.pt"

wandb_config = {
        "project": "triagerx_dt_cv",
        "name": f"run_{output_file}",
        "config": {
        "learning_rate": learning_rate,
        "architecture": "Roberta-CNN",
        "dataset": "deeptriage",
        "epochs": epochs,
    }
}

criterion = CombineLoss(class_weights_tensor)
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

# %%
trainer = ModelTrainer(train_config)
trainer.train(model=model)

# %%
model.load_state_dict(torch.load(output_path))

# %%
from triagerx.evaluation.evaluator import Evaluator

# %%
evaluator = Evaluator()

import numpy as np
# %%
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SimilarityDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizer,
        feature: str = "text",
        target: str = "owner_id",
    ):
        logger.debug("Generating torch dataset...")
        self.tokenizer = tokenizer
        self.labels = [label for label in df[target]]
        logger.debug("Tokenizing texts...")
        self.texts = [
            [row.issue_title, self.tokenizer(
                row[feature],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt",
            )]
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


# %%
from torch.utils.data import DataLoader

from triagerx.dataset.triage_dataset import TriageDataset

dataset = SimilarityDataset(y_df, model.tokenizer())

# %%
len(y_df)

# %%
cuda = torch.cuda.is_available()

if cuda:
    model = model.cuda()

tokenizer = model.tokenizer()

# %%
# triage_dataset = TriageDataset(y_df, model.tokenizer())
loader = DataLoader(dataset, 30)

import numpy as np
# %%
from sentence_transformers import SentenceTransformer, util

# %%
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

# %%
all_embeddings = similarity_model.encode(X_df.issue_title.to_list(), batch_size=15)

# %%
def get_top_k_similar_devs(issues, k=5):
    test_embed = similarity_model.encode(issues)
    cos = util.cos_sim(test_embed, all_embeddings)
    topk = torch.topk(torch.tensor(cos), k=k)
    
    similarities = []
    
    for it in topk.indices.numpy():
        # print(X_df.iloc[it]["owner_id"])
        similarities.append(X_df.iloc[it]["owner_id"].unique().tolist())

    return similarities


# %%
# torch.set_printoptions(sci_mode=True)

# %%
total_acc_val = 0
total_loss_val = 0
correct_top_k = 0
correct_top_k_wo_sim = 0

all_preds = []
all_labels = []
device="cuda"

model = model.cuda()

with torch.no_grad():

    for val_input, val_label in loader:
        val_label = val_label.to(device)
        mask = val_input[1]["attention_mask"].to(device)
        input_id = val_input[1]["input_ids"].squeeze(1).to(device)

        output = model(input_id, mask)



        output = torch.sum(torch.stack(output), 0)

        #wo similarity
        _, top_k_wo_sim = output.topk(10, 1, True, True)

        top_k_wo_sim = top_k_wo_sim.t()

        correct_top_k_wo_sim += (
            top_k_wo_sim.eq(
                val_label.view(1, -1).expand_as(top_k_wo_sim)
            )
            .sum()
            .item()
        )


        # with similarity
        _, top_k_predictions = output.topk(10, 1, True, True)
        similar_preds = get_top_k_similar_devs(val_input[0], k=5)

        unique_preds = []

        for top, sim in zip(top_k_predictions, similar_preds):
            # print(top, sim)
            
            copy_pred = top.cpu().numpy().tolist()
            top_preds = top.cpu().numpy().tolist()[:5]

            for s in sim:
                if s not in top_preds:
                    top_preds.append(s)
            
            if len(top_preds) < 10:
                top_preds = top_preds + copy_pred[5:5 + 10 - len(top_preds)]
            
            unique_preds.append(top_preds)

        unique_preds = torch.tensor(unique_preds).cuda()
        top_k_predictions = unique_preds.t()

        correct_top_k += (
            top_k_predictions.eq(
                val_label.view(1, -1).expand_as(top_k_predictions)
            )
            .sum()
            .item()
        )

        # break

        acc = (output.argmax(dim=1) == val_label).sum().item()

        all_preds.append(output.argmax(dim=1).cpu().numpy())
        all_labels.append(val_label.cpu().numpy())

        total_acc_val += acc

# %%
print(f"Correct Prediction without Similarity: {correct_top_k_wo_sim}")
print(f"Correct Prediction without Similarity: {correct_top_k}")
print(f"Top 10 Acc without Sim: {correct_top_k_wo_sim / len(y_df)}")
print(f"Top 10 Acc with Sim: {correct_top_k / len(y_df)}")


