import torch
from torch import nn
import torch.nn.functional as F
from transformers import RobertaModel, RobertaTokenizer


class RobertaCNNClassifier(nn.Module):
    def __init__(
        self, model_name: str, output_size, embed_size=1024, dropout=0.1
    ) -> None:
        super().__init__()
        self.base_model = RobertaModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        filter_sizes = [3, 4, 5, 6]
        num_filters = 256
        self._tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self._convs = nn.ModuleList(
            [nn.Conv2d(4, num_filters, (K, embed_size)) for K in filter_sizes]
        )
        self._dropout = nn.Dropout(dropout)
        self._fc = nn.Linear(len(filter_sizes) * num_filters, output_size)
        self._relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        x = self.base_model(input_ids, attention_mask=attention_mask)[2][-4:]
        x = torch.stack(x, dim=1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self._convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self._dropout(x)
        logit = self._fc(x)

        return self._relu(logit)

    def tokenizer(self) -> RobertaTokenizer:
        return self._tokenizer
