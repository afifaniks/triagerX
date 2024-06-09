from typing import Dict, List, Optional

import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from triagerx.dataset import EnsembleDataset, TriageDataset
from triagerx.model.cnn_transformer import CNNTransformer
from triagerx.model.prediction_model import PredictionModel
from triagerx.model.triagerx_dev_model import TriagerxDevModel

DEFINED_MODELS = {
    "cnn-transformer": CNNTransformer,
    "triagerx": TriagerxDevModel,
}

DEFINED_DATASETS = {
    CNNTransformer.__name__: TriageDataset,
    TriagerxDevModel.__name__: EnsembleDataset,
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
    ) -> PredictionModel:
        if model_key not in DEFINED_MODELS:
            raise ModuleNotFoundError(model_key, list(DEFINED_MODELS.keys()))

        model_class = DEFINED_MODELS[model_key]

        logger.debug(f"Instantiating model of class: {model_class}")

        model_params = {
            "output_size": output_size,
            "unfrozen_layers": unfrozen_layers,
            "num_classifiers": num_classifiers,
            "dropout": dropout,
            "max_tokens": max_tokens,
            "num_filters": num_filters,
            "label_map": label_map,
        }

        if model_key == "cnn-transformer":
            model_params["base_model"] = base_models[0]
        else:
            model_params["base_models"] = base_models

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
