import torch
from loguru import logger
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from triagerx.model.prediction_model import PredictionModel


class TriagerxDevModel(PredictionModel):
    def __init__(
        self,
        output_size,
        unfrozen_layers,
        num_classifiers,
        base_models,
        dropout=0.1,
        max_tokens=512,
        num_filters=256,
        label_map=None,
    ) -> None:
        super(TriagerxDevModel, self).__init__()

        # Initialize base models and their respective tokenizers
        logger.debug(f"Loading base transformer models: {base_models}")
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

        # Freeze embedding layers for all models
        for base_model in self.base_models:
            for p in base_model.embeddings.parameters():
                p.requires_grad = False

        # Freeze encoder layers until the last `unfrozen_layers` layers for all models
        for base_model in self.base_models:
            for i in range(0, base_model.config.num_hidden_layers - unfrozen_layers):
                for p in base_model.encoder.layer[i].parameters():
                    p.requires_grad = False

        # Define filter sizes for convolution layers
        filter_sizes = [3, 4, 5, 6]
        self._num_filters = num_filters
        self._num_classifiers = num_classifiers
        self._max_tokens = max_tokens
        self.unfrozen_layers = unfrozen_layers

        # Calculate total embedding size by summing hidden sizes of all base models
        embed_sizes = [base_model.config.hidden_size for base_model in self.base_models]
        self._embed_size = sum(embed_sizes)

        # Initialize convolutional blocks for each classifier
        self.conv_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(
                                1,
                                self._num_filters,
                                (K, self._embed_size),
                            ),
                            nn.BatchNorm2d(self._num_filters),
                            nn.ReLU(),
                            nn.Flatten(),
                            nn.MaxPool1d(self._max_tokens - (K - 1)),
                            nn.Flatten(start_dim=1),
                        )
                        for K in filter_sizes
                    ]
                )
                for _ in range(self._num_classifiers)
            ]
        )

        # Initialize classifiers for each classifier block
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(len(filter_sizes) * self._num_filters, output_size)
                for _ in range(self._num_classifiers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize learnable weights for each base model
        self.model_weights = nn.ParameterList(
            [nn.Parameter(torch.ones(self._num_classifiers)) for _ in self.base_models]
        )
        self.classifier_weights = nn.Parameter(torch.ones(self._num_classifiers))

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

        # Concatenate hidden states and apply convolutional blocks and classifiers
        for i in range(self._num_classifiers):
            # Apply learnable weights to hidden states from each base model
            weighted_hidden_states = [
                self.model_weights[idx][i] * hidden_states[idx][i]
                for idx in range(len(self.base_models))
            ]
            concatenated_hidden_states = torch.cat(weighted_hidden_states, dim=-1)
            batch_size, sequence_length, hidden_size = concatenated_hidden_states.size()
            x = [
                conv(
                    concatenated_hidden_states.view(
                        batch_size, 1, sequence_length, hidden_size
                    )
                )
                for conv in self.conv_blocks[i]
            ]
            # Concatenating outputs of the convolutional blocks of different filter sizes
            x = torch.cat(x, dim=1)
            x = self.dropout(x)
            x = self.classifier_weights[i] * self.classifiers[i](x)

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
