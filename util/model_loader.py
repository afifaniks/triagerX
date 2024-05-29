import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer

from triagerx.model.lbt_p_deberta import LBTPDeberta


def get_trained_model(
    weight_path: str,
    num_classes: int,
    unfrozen_layers: int,
    dropout: float,
    base_model: str,
    tokenizer: PreTrainedTokenizer,
) -> nn.Module:
    model = LBTPDeberta(
        output_size=num_classes,
        unfrozen_layers=unfrozen_layers,
        dropout=dropout,
        base_model=base_model,
    )

    model.base_model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(torch.load(weight_path))

    return model
