import torch
from loguru import logger
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from triagerx.model.prediction_model import PredictionModel


class WeightedLayerPooling(nn.Module):
    def __init__(self, num_hidden_layers, layer_start: int = 4, layer_weights=None):
        super(WeightedLayerPooling, self).__init__()
        self.layer_start = layer_start
        self.num_hidden_layers = num_hidden_layers
        self.layer_weights = (
            layer_weights
            if layer_weights is not None
            else nn.Parameter(
                torch.tensor(
                    [1] * (num_hidden_layers + 1 - layer_start), dtype=torch.float
                )
            )
        )

    def forward(self, all_hidden_states):
        all_layer_embedding = all_hidden_states[self.layer_start :, :, :, :]
        weight_factor = (
            self.layer_weights.unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .expand(all_layer_embedding.size())
        )
        weighted_average = (weight_factor * all_layer_embedding).sum(
            dim=0
        ) / self.layer_weights.sum()
        return weighted_average


class TriagerxFCNPoolerModel(PredictionModel):
    def __init__(
        self,
        output_size,
        unfrozen_layers,
        num_classifiers,
        base_models,
        dropout=0.1,
        max_tokens=512,
        label_map=None,
    ) -> None:
        super(TriagerxFCNPoolerModel, self).__init__()

        # Initialize base models and their respective tokenizers
        logger.debug(f"Loading TriagerxFCN base transformer models: {base_models}")
        self.base_models = nn.ModuleList(
            [
                AutoModel.from_pretrained(model, output_hidden_states=True)
                for model in base_models
            ]
        )
        self.tokenizers = [
            AutoTokenizer.from_pretrained(model) for model in base_models
        ]
        self._label_map = label_map

        if unfrozen_layers == -1:
            logger.debug("Initiating full training...")
        else:
            # Freeze embedding layers for all models
            for base_model in self.base_models:
                logger.debug(
                    f"Freezing {base_model.config.num_hidden_layers - unfrozen_layers} layers"
                )
                for p in base_model.embeddings.parameters():
                    p.requires_grad = False

            # Freeze encoder layers until the last `unfrozen_layers` layers for all models
            for base_model in self.base_models:
                for i in range(
                    0, base_model.config.num_hidden_layers - unfrozen_layers
                ):
                    for p in base_model.encoder.layer[i].parameters():
                        p.requires_grad = False

        self._num_classifiers = num_classifiers
        self._max_tokens = max_tokens
        self.unfrozen_layers = unfrozen_layers

        self._poolers = nn.ModuleList(
            [
                WeightedLayerPooling(
                    base_model.config.num_hidden_layers,
                    layer_start=6,
                    layer_weights=None,
                )
                for base_model in base_models
            ]
        )

        # Calculate total embedding size by summing hidden sizes of all base models
        embed_sizes = [base_model.config.hidden_size for base_model in self.base_models]
        self._embed_size = sum(embed_sizes)

        # Initialize classifiers for each classifier block
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._embed_size, self._embed_size),
            nn.ReLU(),
            nn.Linear(self._embed_size, output_size),
        )

        # Initialize learnable weights for each base model
        # self.model_weights = nn.ParameterList(
        #     [nn.Parameter(torch.ones(self._num_classifiers)) for _ in self.base_models]
        # )
        # self.classifier_weights = nn.Parameter(torch.ones(self._num_classifiers))

    def forward(self, inputs):
        # Process input data for each base model
        inputs = [
            {
                key: value.squeeze(1).to(next(self.parameters()).device)
                for key, value in model_inputs.items()
            }
            for model_inputs in inputs
        ]

        hidden_states = []

        # Extract hidden states from each base model
        for idx, base_model in enumerate(self.base_models):
            base_out = base_model(
                input_ids=inputs[idx]["input_ids"],
                attention_mask=inputs[idx]["attention_mask"],
            )
            hidden_states.append(base_out.hidden_states[-self._num_classifiers :])

        outputs = []

        # Concatenate hidden states and apply classifiers
        for i in range(self._num_classifiers):
            # Apply learnable weights to hidden states from each base model
            pooled_hidden_states = [
                self._poolers(hidden_states[idx])
                for idx in range(len(self.base_models))
            ]
            concatenated_hidden_states = torch.cat(pooled_hidden_states, dim=-1)
            x = concatenated_hidden_states
            x = torch.mean(concatenated_hidden_states, dim=1)
            x = self.classifiers[i](x)

            outputs.append(x)

        return outputs

    def tokenizer(self, model_idx) -> PreTrainedTokenizer:
        # Return the tokenizer for the specified base model index
        if model_idx < 0 or model_idx >= len(self.tokenizers):
            raise ValueError(
                f"Invalid model index, choose between 0 and {len(self.tokenizers) - 1}"
            )
        return self.tokenizers[model_idx]

    def tokenize_text(self, text):
        return [
            tokenizer(
                text,
                padding="max_length",
                max_length=self._max_tokens,
                truncation=True,
                return_tensors="pt",
            )
            for tokenizer in self.tokenizers
        ]

    def get_label_map(self):
        return self._label_map
