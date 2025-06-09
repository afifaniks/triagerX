import itertools
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.append("/home/mdafifal.mamun/notebooks/triagerX/")

from triagerx.model.module_factory import ModelFactory
from triagerx.system.triagerx import TriagerX

tqdm.pandas()

torch.manual_seed(42)

target_components = [
    "comp:vm",
    "comp:jvmti",
    "comp:jclextensions",
    "comp:test",
    "comp:build",
    "comp:gc",
]
target_components = sorted(target_components)


# TS
df_train = pd.read_csv(
    "/home/mdafifal.mamun/notebooks/triagerX/data/typescript/ts_train.csv"
)
df_test = pd.read_csv(
    "/home/mdafifal.mamun/notebooks/triagerX/data/typescript/ts_test.csv"
)
output_file = "/home/mdafifal.mamun/notebooks/triagerX/grid_reports/ts_grid_search_50_final_sim_threshold.csv"
developer_model_weights = "/work/disa_lab/projects/triagerx/models/typescript/ts_triagerx_ensemble_u3_40_classes_last_dev_seed42.pt"
component_model_weights = "/work/disa_lab/projects/triagerx/models/openj9/component_deberta-base_u3_6_classes_seed42.pt"
train_embeddings_path = (
    "/home/mdafifal.mamun/notebooks/triagerX/data/typescript/embeddings_40devs.npy"
)
MAX_K = 20

# PAPER CONFIG
# df_train = pd.read_csv(
#     "/home/mdafifal.mamun/notebooks/triagerX/old_data/openj9/last_contribution/openj9_train.csv"
# )
# df_test = pd.read_csv(
#     "/home/mdafifal.mamun/notebooks/triagerX/old_data/openj9/last_contribution/openj9_test.csv"
# )
# output_file = (
#     "/home/mdafifal.mamun/notebooks/triagerX/grid_reports/grid_search_50_final.csv"
# )
# developer_model_weights = "/work/disa_lab/projects/triagerx/models/openj9/triagerx_ensemble_u3_50_classes_last_dev_seed42.pt"
# component_model_weights = "/work/disa_lab/projects/triagerx/models/openj9/component_deberta-base_u3_6_classes_seed42.pt"
# train_embeddings_path = (
#     "/home/mdafifal.mamun/notebooks/triagerX/data/openj9/embeddings_50devs.npy"
# )
# MAX_K = 20

# IBM CONFIG
# df_train = pd.read_csv("/home/mdafifal.mamun/notebooks/triagerX/openj9_train_17.csv")
# df_test = pd.read_csv("/home/mdafifal.mamun/notebooks/triagerX/openj9_test_17.csv")
# output_file = "/home/mdafifal.mamun/notebooks/triagerX/grid_reports/grid_search_ibm.csv"
# developer_model_weights = "/home/mdafifal.mamun/notebooks/triagerX/app/saved_states/triagerx_ensemble_u3_last_devs_17devs_seed42.pt"
# component_model_weights = "/home/mdafifal.mamun/notebooks/triagerX/app/saved_states/component_deberta-base_u3_6_classes_seed42.pt"
# train_embeddings_path = (
#     "/home/mdafifal.mamun/notebooks/triagerX/app/saved_states/train_embeddings.npy"
# )
# MAX_K = 15

df_train["owner"] = df_train.owner.apply(lambda x: x.lower())
df_test["owner"] = df_test.owner.apply(lambda x: x.lower())

logger.info(f"Training data: {len(df_train)}, Validation data: {len(df_test)}")
logger.info(f"Number of train developers: {len(df_train.owner.unique())}")
logger.info(f"Number of test developers: {len(df_test.owner.unique())}")

logger.info(f"Train dataset size: {len(df_train)}")
logger.info(f"Test dataset size: {len(df_test)}")


lbl2idx = dict(zip(df_train["owner"], df_train["owner_id"]))
idx2lbl = dict(zip(df_train["owner_id"], df_train["owner"]))

# df_train = df_train[df_train["component"].notna()]
# df_test = df_test[df_test["component"].notna()]


comp_id2label = {}
comp_lbl2id = {}

for i, comp in enumerate(target_components):
    comp_id2label[i] = comp
    comp_lbl2id[comp] = i

base_transformer_models = ["microsoft/deberta-base", "roberta-base"]
device = "cuda" if torch.cuda.is_available() else "cpu"
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

dev_model.load_state_dict(
    torch.load(
        developer_model_weights,
        map_location=device,
    )
)


comp_model = ModelFactory.get_model(
    model_key="cnn-transformer",
    output_size=6,
    unfrozen_layers=3,
    num_classifiers=3,
    base_models=["microsoft/deberta-base"],
    dropout=0.2,
    max_tokens=256,
    label_map=comp_id2label,
)
comp_model.load_state_dict(
    torch.load(
        component_model_weights,
        map_location=device,
    )
)


logger.debug("Generating embeddings...")
similarity_model = SentenceTransformer("all-mpnet-base-v2")

if not os.path.exists(train_embeddings_path):
    logger.debug("Embedding doesn't exist, generating new...")
    encodings = similarity_model.encode(df_train.text.tolist(), show_progress_bar=True)
    np.save(train_embeddings_path, encodings)


def get_recommendation(trx, test_idx, k_comp, k_dev, k_rank, sim):
    test_data = df_test.iloc[test_idx]

    return trx.get_recommendation(
        test_data.text,
        k_comp=k_comp,
        k_dev=k_dev,
        k_rank=k_rank,
        similarity_threshold=sim,
    )


def get_topk_score(recommendations, top_k):
    combined_total = 0
    dl_total = 0
    sim_total = 0
    borda_total = 0

    for idx in range(len(df_test)):
        actual = df_test.iloc[idx]["owner"]
        combined_recommended = recommendations[idx]["combined_ranking"][:top_k]
        dl_recommended = recommendations[idx]["predicted_developers"][:top_k]
        sim_recommended = recommendations[idx]["similar_devs"][:top_k]
        borda_recommended = recommendations[idx]["borda_ranking"][:top_k]

        if actual in combined_recommended:
            combined_total += 1

        if actual in dl_recommended:
            dl_total += 1

        if actual in sim_recommended:
            sim_total += 1

        if actual in borda_recommended:
            borda_total += 1

    return (
        dl_total / len(df_test),
        sim_total / len(df_test),
        combined_total / len(df_test),
        borda_total / len(df_test),
    )


def evaluate_recommendations(params):
    # Extract parameters
    similarity_prediction_weight = params["similarity_prediction_weight"]
    time_decay_factor = params["time_decay_factor"]
    direct_assignment_score = params["direct_assignment_score"]
    contribution_score = params["contribution_score"]
    discussion_score = params["discussion_score"]
    similarity_threshold = params["similarity_threshold"]

    expected_users = set(df_train.owner.unique())
    print(expected_users)

    trx = TriagerX(
        component_prediction_model=comp_model,
        developer_prediction_model=dev_model,
        similarity_model=similarity_model,
        issues_path="/home/mdafifal.mamun/notebooks/triagerX/data/typescript/issue_data",
        train_embeddings=train_embeddings_path,
        developer_id_map=lbl2idx,
        component_id_map=comp_lbl2id,
        expected_developers=expected_users,
        train_data=df_train,
        device=device,
        similarity_prediction_weight=similarity_prediction_weight,
        time_decay_factor=time_decay_factor,
        direct_assignment_score=direct_assignment_score,
        contribution_score=contribution_score,
        discussion_score=discussion_score,
        train_checkpoint_date=datetime.strptime("2024-06-27", "%Y-%m-%d"),
    )

    recommendations = []

    for i in tqdm(range(len(df_test)), total=len(df_test), desc="Processing..."):
        rec = get_recommendation(
            trx, i, k_comp=3, k_dev=MAX_K, k_rank=20, sim=similarity_threshold
        )
        recommendations.append(rec)

    top_1 = get_topk_score(recommendations, 1)
    top_3 = get_topk_score(recommendations, 3)
    top_5 = get_topk_score(recommendations, 5)
    top_10 = get_topk_score(recommendations, 10)
    top_20 = get_topk_score(recommendations, 20)

    return top_1, top_3, top_5, top_10, top_20


# parameter_ranges = {
#     "similarity_prediction_weight": [0.5, 0.6, 0.7],
#     "time_decay_factor": [0.01, 0.03, 0.05],
#     "direct_assignment_score": [1.0, 1.5, 2.0],
#     "contribution_score": [1.0, 1.5, 2.0],
#     "discussion_score": [0.5, 1.0],
#     "similarity_threshold": [0.5, 0.6, 0.65, 0.7],
# }

parameter_ranges = {
    "similarity_prediction_weight": [0.25],
    "time_decay_factor": [0.001],
    "direct_assignment_score": [0.5],
    "contribution_score": [1.5],
    "discussion_score": [0.1],
    "similarity_threshold": np.arange(0, 1.01, 0.05),
}

total_combinations = len(list(itertools.product(*parameter_ranges.values())))

# Initialize an empty list to store results
results = []

index = 1
# Iterate over all combinations
for params in itertools.product(*parameter_ranges.values()):

    print(f"Running Grid Search... {index}/{total_combinations}")
    index += 1

    params_dict = {
        "similarity_prediction_weight": params[0],
        "time_decay_factor": params[1],
        "direct_assignment_score": params[2],
        "contribution_score": params[3],
        "discussion_score": params[4],
        "similarity_threshold": params[5],
    }

    top_1, top_3, top_5, top_10, top_20 = evaluate_recommendations(params_dict)

    # Append results to the list
    result = {
        "similarity_prediction_weight": params_dict["similarity_prediction_weight"],
        "time_decay_factor": params_dict["time_decay_factor"],
        "direct_assignment_score": params_dict["direct_assignment_score"],
        "contribution_score": params_dict["contribution_score"],
        "discussion_score": params_dict["discussion_score"],
        "similarity_threshold": params_dict["similarity_threshold"],
        "T1DL": top_1[0],
        "T1Sim": top_1[1],
        "T1Com": top_1[2],
        "T1Borda": top_1[3],
        "T3DL": top_3[0],
        "T3Sim": top_3[1],
        "T3Com": top_3[2],
        "T3Borda": top_3[3],
        "T5DL": top_5[0],
        "T5Sim": top_5[1],
        "T5Com": top_5[2],
        "T5Borda": top_5[3],
        "T10DL": top_10[0],
        "T10Sim": top_10[1],
        "T10Com": top_10[2],
        "T10Borda": top_10[3],
        "T20DL": top_20[0],
        "T20Sim": top_20[1],
        "T20Com": top_20[2],
        "T20Borda": top_20[3],
    }

    logger.info(f"Result: {result}")
    results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)

    print(f"Grid search results saved to {output_file}")

df = pd.DataFrame(results)

# Write DataFrame to CSV file
df.to_csv(output_file, index=False)

print(f"Grid search results saved to {output_file}")
