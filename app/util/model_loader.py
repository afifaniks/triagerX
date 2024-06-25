from typing import Dict

import torch

from triagerx.model.module_factory import ModelFactory
from triagerx.model.prediction_model import PredictionModel


def get_trained_model(model_config: Dict) -> PredictionModel:
    model = ModelFactory.get_model(**model_config["config"])
    model.load_state_dict(torch.load(model_config["weights_path"]))

    return model
