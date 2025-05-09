import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, PreTrainedTokenizer, RobertaModel


class LBTPClassifier(nn.Module):
    def __init__(
        self,
        embedding_model,
        output_size,
        unfrozen_layers=1,
        num_classifiers=3,
        max_tokens=256,
    ) -> None:
        super().__init__()
        self.base_model = embedding_model

        # Freeze embedding layers
        for p in self.base_model.embeddings.parameters():
            p.requires_grad = False

        # Freeze encoder layers till last {unfrozen_layers} layers
        for i in range(0, self.base_model.config.num_hidden_layers - unfrozen_layers):
            for p in self.base_model.encoder.layer[i].parameters():
                p.requires_grad = False

        filter_sizes = [3, 4, 5, 6]
        self._num_filters = 256
        self._max_tokens = max_tokens
        self._num_classifiers = num_classifiers
        self._embed_size = embedding_model.config.hidden_size
        self.unfrozen_layers = unfrozen_layers
        self._tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.conv_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(1, self._num_filters, (K, self._embed_size)),
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

        self.classifier_weights = nn.Parameter(torch.ones(self._num_classifiers))

        self.classifiers = nn.ModuleList(
            [
                nn.Linear(
                    len(filter_sizes) * self._num_filters + self._embed_size,
                    output_size,
                )
                for _ in range(self._num_classifiers)
            ]
        )

        # Dropout is ommitted as it is not mentioned in the LBTP paper
        # self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        input_ids = input["input_ids"].squeeze(1).to(self._device)
        attention_mask = input["attention_mask"].squeeze(1).to(self._device)

        outputs = []

        base_out = self.base_model(input_ids, attention_mask=attention_mask)
        pooler_out = base_out.pooler_output.squeeze(0)
        hidden_states = base_out.hidden_states[-self._num_classifiers :]

        for i in range(self._num_classifiers):
            batch_size, sequence_length, hidden_size = hidden_states[i].size()
            x = [
                conv(hidden_states[i].view(batch_size, 1, sequence_length, hidden_size))
                for conv in self.conv_blocks[i]
            ]
            x = torch.cat(x, dim=1)
            x = torch.cat([pooler_out, x], dim=1)
            x = self.classifier_weights[i] * self.classifiers[i](x)

            outputs.append(x)

        return outputs

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
