from typing import Any, Dict

import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from triagerx.dataset.text_processor import TextProcessor
from triagerx.system.triagerx import TriagerX
from util.model_loader import get_trained_model


class RecommendationService:
    def __init__(self, config: Dict[Any, Any]):
        self._config = config
        logger.debug("Loading pretrained weights...")
        self._component_model = self._load_trained_model(config["component_model"])
        self._developer_model = self._load_trained_model(config["developer_model"])
        self._similarity_model = SentenceTransformer(
            config["similarity_model"]["model_key"]
        )

        tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"]["model_name"])
        tokenizer_config = config["tokenizer"]
        developer_id_map = config["developer_id_map"]
        component_id_map = config["component_id_map"]
        component_dev_map = config["component_developer_map"]
        train_df = pd.read_csv(config["data"]["train_data"])
        issues_dir = config["data"]["issues_dir"]

        logger.debug("Initializing Triager X system...")
        self.triager = TriagerX(
            developer_prediction_model=self._developer_model,
            component_prediction_model=self._component_model,
            similarity_model=self._similarity_model,
            tokenizer=tokenizer,
            tokenizer_config=tokenizer_config,
            train_data=train_df,
            developer_id_map=developer_id_map,
            component_id_map=component_id_map,
            component_developers_map=component_dev_map,
            issues_path=issues_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def _load_trained_model(self, model_config: Dict):
        logger.debug(
            f"Loading pretrained model: {model_config['weights_save_location']}"
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["base_transformer_model"]
        )
        return get_trained_model(
            weight_path=model_config["weights_save_location"],
            num_classes=model_config["num_classes"],
            unfrozen_layers=model_config["unfrozen_layers"],
            dropout=model_config["dropout"],
            base_model=model_config["base_transformer_model"],
            tokenizer=tokenizer,
        )

    def get_recommendation(self, issue_title: str, issue_description: str):
        processed_issue = TextProcessor.prepare_text(
            issue_title,
            issue_description,
            summary="",
            use_description=True,
            use_summary=False,
            use_special_tokens=False,
        )
        return self.triager.get_recommendation(
            processed_issue, k_comp=3, k_dev=3, k_rank=5, similarity_threshold=0.5
        )
