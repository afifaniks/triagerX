import os
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer

from app.util.model_loader import get_trained_model
from triagerx.dataset.text_processor import TextProcessor
from triagerx.system.triagerx import TriagerX


class RecommendationService:
    def __init__(self, config: Dict[Any, Any]):
        self._config = config
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.debug(f"Using device: {self._device}")

        self._initialize_models()
        self._initialize_triager()

    def _initialize_models(self):
        logger.debug("Loading pretrained weights...")
        self._developer_model = self._load_trained_model(
            self._config["developer_model"]
        )
        self._similarity_model = SentenceTransformer(
            self._config["similarity_model"]["model_key"]
        )

        embeddings_path = self._config["similarity_model"]["embeddings_path"]
        if not os.path.exists(embeddings_path):
            logger.debug("Historical issue embedding doesn't exist, generating new...")
            df_train = pd.read_csv(self._config["data"]["train_data"])
            encodings = self._similarity_model.encode(
                df_train.text.tolist(), show_progress_bar=True
            )
            np.save(embeddings_path, encodings)

    def _initialize_triager(self):
        logger.debug("Initializing Triager X engine...")
        train_data = pd.read_csv(self._config["data"]["train_data"])
        developer_id_map = pd.Series(
            train_data["owner_id"].values, index=train_data["owner"]
        ).to_dict()

        self.triager = TriagerX(
            developer_prediction_model=self._developer_model,
            similarity_model=self._similarity_model,
            train_data=train_data,
            train_embeddings=self._config["similarity_model"]["embeddings_path"],
            issues_path=self._config["data"]["issues_path"],
            developer_id_map=developer_id_map,
            expected_developers=set(developer_id_map.keys()),
            device=self._device,
            similarity_prediction_weight=self._config["contribution_score_params"][
                "similarity_prediction_weight"
            ],
            time_decay_factor=self._config["contribution_score_params"][
                "time_decay_factor"
            ],
            direct_assignment_score=self._config["contribution_score_params"][
                "direct_assignment_score"
            ],
            contribution_score=self._config["contribution_score_params"][
                "contribution_score"
            ],
            discussion_score=self._config["contribution_score_params"][
                "discussion_score"
            ],
        )

    def _load_trained_model(self, model_config: Dict):
        logger.debug(f"Loading pretrained model: {model_config['weights_path']}")
        return get_trained_model(model_config, self._device)

    def get_recommendation(self, issue_title: str, issue_description: str):
        processed_issue = TextProcessor.prepare_text(issue_title, issue_description)
        return self.triager.get_recommendation(
            processed_issue,
            k_dev=3,
            k_rank=self._config["contribution_score_params"]["maximum_similar_issues"],
            similarity_threshold=self._config["contribution_score_params"][
                "similarity_threshold"
            ],
        )
