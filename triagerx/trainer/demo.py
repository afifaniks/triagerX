import itertools
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append("../triagerX/")

from triagerx.model.module_factory import ModelFactory
from triagerx.system.triagerx import TriagerX

tqdm.pandas()

torch.manual_seed(42)


# PAPER CONFIG
df_train = pd.read_csv("./data/openj9/openj9_train.csv")
df_test = pd.read_csv("./data/openj9/openj9_test.csv")
train_embeddings_path = "./data/openj9/embeddings_50devs.npy"
developer_model_weights = "/work/disa_lab/projects/triagerx/models/openj9/triagerx_ensemble_u3_50_classes_last_dev_seed42.pt"

df_train["owner"] = df_train.owner.apply(lambda x: x.lower())
df_test["owner"] = df_test.owner.apply(lambda x: x.lower())

logger.info(f"Train dataset size: {len(df_train)}")
logger.info(f"Test dataset size: {len(df_test)}")

lbl2idx = dict(zip(df_train["owner"], df_train["owner_id"]))
idx2lbl = dict(zip(df_train["owner_id"], df_train["owner"]))

base_transformer_models = ["microsoft/deberta-base", "roberta-base"]

logger.debug("Modeling network...")
dev_model = ModelFactory.get_model(
    model_key="triagerx",
    output_size=len(df_train.owner_id.unique()),
    unfrozen_layers=3,
    num_classifiers=3,
    base_models=base_transformer_models,
    dropout=0.2,
    max_tokens=256,
    label_map=idx2lbl,
)

dev_model.load_state_dict(torch.load(developer_model_weights))

logger.debug("Generating embeddings...")
similarity_model = SentenceTransformer("all-mpnet-base-v2")

if not os.path.exists(train_embeddings_path):
    logger.debug("Embedding doesn't exist, generating new...")
    encodings = similarity_model.encode(df_train.text.tolist(), show_progress_bar=True)
    np.save(train_embeddings_path, encodings)

expected_users = set(df_train.owner.unique())

SIM_PREDICTION_WEIGHT = 0.65
TIME_DECAY_FACTOR = 0.01
DIRECT_ASSIGNMENT_SCORE = 0.5
CONTRIBUTION_SCORE = 1.5
DISCUSSION_SCORE = 0.2
SIMILAITY_THRESHOLD = 0.5
MAX_K = 20

trx = TriagerX(
    developer_prediction_model=dev_model,
    similarity_model=similarity_model,
    issues_path="./data/openj9/issue_data",
    train_embeddings=train_embeddings_path,
    developer_id_map=lbl2idx,
    expected_developers=expected_users,
    train_data=df_train,
    device="cuda",
    similarity_prediction_weight=SIM_PREDICTION_WEIGHT,
    time_decay_factor=TIME_DECAY_FACTOR,
    direct_assignment_score=DIRECT_ASSIGNMENT_SCORE,
    contribution_score=CONTRIBUTION_SCORE,
    discussion_score=DISCUSSION_SCORE,
)

random.seed(42)
test_index = random.randint(0, len(df_test))

recommendation = trx.get_recommendation(
    df_test.iloc[test_index].text,
    k_dev=3,
    k_rank=MAX_K,
    similarity_threshold=SIMILAITY_THRESHOLD,
)

print(f"Actual Owner: {df_test.iloc[test_index].owner}")
print(f"Recommendations: {recommendation['combined_ranking']}")
