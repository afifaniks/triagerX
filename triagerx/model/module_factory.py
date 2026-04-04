from typing import Dict, List, Optional

import pandas as pd
import torch.nn as nn
from loguru import logger
from torch.utils.data import Dataset
from transformers import RobertaConfig, RobertaModel

from triagerx.dataset import EnsembleDataset, TriageDataset
from triagerx.model.cnn_transformer import CNNTransformer
from triagerx.model.fcn_transformer import FCNTransformer
from triagerx.model.lbt_p import LBTPClassifier
from triagerx.model.prediction_model import PredictionModel
from triagerx.model.triagerx_model import TriagerxModel
from triagerx.model.triagerx_pooler import TriagerxFCNPoolerModel
from triagerx.model.triagerx_sequential import TriagerxFCNModel

DEFINED_MODELS = {
    "cnn-transformer": CNNTransformer,
    "fcn-transformer": FCNTransformer,
    "triagerx": TriagerxModel,
    "triagerx-fcn": TriagerxFCNModel,
    "triagerx-fcn-pooler": TriagerxFCNPoolerModel,
    "lbtp": LBTPClassifier,
}

DEFINED_DATASETS = {
    CNNTransformer.__name__: TriageDataset,
    FCNTransformer.__name__: TriageDataset,
    TriagerxModel.__name__: EnsembleDataset,
    TriagerxFCNModel.__name__: EnsembleDataset,
    TriagerxFCNPoolerModel.__name__: EnsembleDataset,
    LBTPClassifier.__name__: TriageDataset,
}


class ModuleNotFoundError(Exception):
    """Custom exception for handling model not found errors."""

    def __init__(self, model_key: str, defined_models: List[str]):
        message = (
            f"Module key '{model_key}' not found. Accepted modules: {defined_models}"
        )
        super().__init__(message)


class ModelFactory:
    @staticmethod
    def get_model(
        model_key: str,
        output_size: int,
        unfrozen_layers: int,
        num_classifiers: int,
        base_models: List[str],
        dropout: float = 0.1,
        max_tokens: int = 512,
        num_filters: int = 256,
        label_map: Optional[Dict[int, str]] = None,
    ) -> nn.Module:
        if model_key not in DEFINED_MODELS:
            raise ModuleNotFoundError(model_key, list(DEFINED_MODELS.keys()))

        if model_key == "lbtp":
            logger.debug("Using LBT-P model...")
            model_config = RobertaConfig.from_pretrained("roberta-large")
            model_config.num_hidden_layers = 3
            model_config.output_hidden_states = True
            embedding_model = RobertaModel(model_config)

            return LBTPClassifier(
                embedding_model, output_size=output_size, unfrozen_layers=3
            )

        model_class = DEFINED_MODELS[model_key]

        logger.debug(f"Instantiating model of class: {model_class}")

        model_params = {
            "output_size": output_size,
            "unfrozen_layers": unfrozen_layers,
            "dropout": dropout,
            "max_tokens": max_tokens,
            "label_map": label_map,
        }

        if "fcn" not in model_key:
            logger.debug("Including number of filters and classifiers")
            model_params["num_classifiers"] = num_classifiers
            model_params["num_filters"] = num_filters
        else:
            logger.debug("Ignoring number of filters and classifiers for FCN")

        if "triagerx" in model_key:
            model_params["num_classifiers"] = num_classifiers
            model_params["base_models"] = base_models
        else:
            model_params["base_model"] = base_models[0]

        return model_class(**model_params)


class DatasetFactory:
    @staticmethod
    def get_dataset(
        df: pd.DataFrame,
        model: PredictionModel,
        feature_field: str,
        target_field: str,
        max_length: int,
    ) -> Dataset:
        model_name = model.__class__.__name__

        if model_name not in DEFINED_DATASETS:
            raise ModuleNotFoundError(model_name, list(DEFINED_DATASETS.keys()))

        dataset_class = DEFINED_DATASETS[model_name]

        return dataset_class(
            df=df,
            model=model,
            feature=feature_field,
            target=target_field,
            max_length=max_length,
        )
