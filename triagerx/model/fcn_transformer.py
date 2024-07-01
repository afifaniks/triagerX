import torch
from loguru import logger
from torch import nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer

from triagerx.model.prediction_model import PredictionModel


class FCNTransformer(PredictionModel):
    def __init__(
        self,
        output_size,
        unfrozen_layers,
        dropout=0.1,
        base_model="microsoft/deberta-large",
        max_tokens=512,
        label_map=None,
    ) -> None:
        super(FCNTransformer, self).__init__()
        self.base_model = AutoModel.from_pretrained(
            base_model, output_hidden_states=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        self._label_map = label_map

        if unfrozen_layers == -1:
            logger.debug("Initiating full training...")
        else:
            logger.debug(
                f"Freezing {self.base_model.config.num_hidden_layers - unfrozen_layers} layers"
            )
            # Freeze embedding layers
            for p in self.base_model.embeddings.parameters():
                p.requires_grad = False

            # Freeze encoder layers till last {unfrozen_layers} layers
            for i in range(
                0, self.base_model.config.num_hidden_layers - unfrozen_layers
            ):
                for p in self.base_model.encoder.layer[i].parameters():
                    p.requires_grad = False

        self._max_tokens = max_tokens
        self._embed_size = self.base_model.config.hidden_size
        self.unfrozen_layers = unfrozen_layers

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self._embed_size * self._max_tokens, output_size)

    def forward(self, inputs):
        inputs = {
            key: value.squeeze(1).to(next(self.parameters()).device)
            for key, value in inputs.items()
        }

        base_out = self.base_model(**inputs)
        hidden_state = base_out.hidden_states[-1]

        batch_size, sequence_length, hidden_size = hidden_state.size()
        x = hidden_state.view(batch_size, -1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer

    def tokenize_text(self, text):
        return self._tokenizer(
            text,
            padding="max_length",
            max_length=self._max_tokens,
            truncation=True,
            return_tensors="pt",
        )

    def get_label_map(self):
        return self._label_map
