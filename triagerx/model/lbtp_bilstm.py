import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


class LBTPBiLSTM(nn.Module):
    def __init__(
        self, output_size, unfrozen_layers=4, dropout=0.1, base_model="microsoft/deberta-large"
    ) -> None:
        super().__init__()
        self.base_model = AutoModel.from_pretrained(
            base_model, output_hidden_states=True
        )
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)        

        # Freeze embedding layers
        for p in self.base_model.embeddings.parameters():
            p.requires_grad = False

        # Freeze encoder layers till last {unfrozen_layers} layers
        for i in range(0, self.base_model.config.num_hidden_layers - unfrozen_layers):
            for p in self.base_model.encoder.layer[i].parameters():
                p.requires_grad = False

        filter_sizes = [3, 4, 5, 6]
        self._num_filters = 256
        self._max_tokens = 512
        self._embed_size = self.base_model.config.hidden_size
        self.unfrozen_layers = unfrozen_layers
        self.conv_blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Conv2d(1, self._num_filters, (K, self._embed_size)),
                            nn.BatchNorm2d(self._num_filters),
                            nn.ReLU(),
                            nn.Flatten(),
                            nn.MaxPool1d(self._max_tokens - (K - 1)),
                            nn.Flatten(start_dim=1),
                        )
                        for K in filter_sizes
                    ]
                )
                for _ in range(unfrozen_layers)
            ]
        )

        self.lstm = nn.ModuleList(
            [
                nn.LSTM(
                    len(filter_sizes) * self._num_filters, 
                    num_layers=3, 
                    hidden_size=256, 
                    bidirectional=True, 
                    batch_first=True
                )
                for _ in range(unfrozen_layers)
            ]
        )

        self.classifiers = nn.ModuleList(
            [
                nn.Linear(
                    512, output_size
                )
                for _ in range(unfrozen_layers)
            ]
        )
        
        self.classifier_weights = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(unfrozen_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, tok_type):
        outputs = []

        base_out = self.base_model(input_ids=input_ids, token_type_ids=tok_type, attention_mask=attention_mask)
        # pooler_out = base_out.last_hidden_state.squeeze(0)
        hidden_states = base_out.hidden_states[-self.unfrozen_layers :]

        for i in range(self.unfrozen_layers):
            batch_size, sequence_length, hidden_size = hidden_states[i].size()
            x = [
                conv(hidden_states[i].view(batch_size, 1, sequence_length, hidden_size))
                for conv in self.conv_blocks[i]
            ]
            # Concatanating outputs of the conv block of different filter sizes
            x = torch.cat(x, dim=1)
            x = self.dropout(x)
            x, _ = self.lstm[i](x.unsqueeze(1))
            x = self.dropout(x)
            x = self.classifiers[i](x[:, -1, :]) * self.classifier_weights[i]

            outputs.append(x)

        return outputs

    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer
