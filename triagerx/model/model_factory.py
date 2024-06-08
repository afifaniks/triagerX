from typing import Dict, List, Optional

from loguru import logger

from triagerx.model.cnn_transformer import CNNTransformer
from triagerx.model.prediction_model import PredictionModel
from triagerx.model.triagerx_dev_model import TriagerxDevModel


class ModelNotFoundError(Exception):
    """Custom exception for handling model not found errors."""

    def __init__(self, model_key: str, defined_models: List[str]):
        message = (
            f"Model key '{model_key}' not found. Accepted models: {defined_models}"
        )
        super().__init__(message)


class ModelFactory:
    DEFINED_MODELS = {
        "cnn-transformer": CNNTransformer,
        "triagerx": TriagerxDevModel,
    }

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
        if model_key not in ModelFactory.DEFINED_MODELS:
            raise ModelNotFoundError(
                model_key, list(ModelFactory.DEFINED_MODELS.keys())
            )

        model_class = ModelFactory.DEFINED_MODELS[model_key]

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
