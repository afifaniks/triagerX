import torch
import torch.nn.functional as F
from torch import nn
from transformers import RobertaModel, RobertaTokenizer


class LBTPClassifier(nn.Module):
    def __init__(
        self, output_size, bert_unfreeze_layers=4, embed_size=1024, dropout=0.1
    ) -> None:
        super().__init__()
        model_name = "roberta-large"
        self.base_model = RobertaModel.from_pretrained(
            model_name, output_hidden_states=True
        )
        self._tokenizer = RobertaTokenizer.from_pretrained(model_name)

        # Freeze embedding layers
        for p in self.base_model.embeddings.parameters():
            p.requires_grad = False

        # Freeze encoder layers till defined unfreeze layers
        for i in range (0, 24 - bert_unfreeze_layers):
            for p in self.base_model.encoder.layer[i].parameters():
                p.requires_grad = False

        filter_sizes = [3, 4, 5, 6]
        self._num_filters = 256
        self._max_tokens = 512
        self._embed_size = embed_size
        self._bert_unfreeze_layers = bert_unfreeze_layers
        self.conv_blocks = nn.ModuleList(
            [nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(1, self._num_filters, (K, embed_size)),
                nn.ReLU(),
                nn.Flatten(),
                nn.MaxPool1d(self._max_tokens - (K - 1)),
                nn.Flatten(start_dim=1)
            )
            for K in filter_sizes])
            for _ in range(bert_unfreeze_layers)]
        ) 

        self.classifiers = nn.ModuleList([
            nn.Linear(len(filter_sizes) * self._num_filters + embed_size, output_size)
            for _ in range(bert_unfreeze_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        outputs = []

        base_out = self.base_model(input_ids, attention_mask=attention_mask)
        pooler_out = base_out.pooler_output.squeeze(0)
        hidden_states = base_out.hidden_states[-self._bert_unfreeze_layers:]

        for i in range(self._bert_unfreeze_layers):
          batch_size, sequence_length, hidden_size = hidden_states[i].size()
          x = [conv(hidden_states[i].view(batch_size, 1, sequence_length, hidden_size)) for conv in self.conv_blocks[i]]          
          x = torch.cat(x, dim=1)   
          x = torch.cat([pooler_out, x], dim=1)    
          x = self.dropout(x)
          x = self.classifiers[i](x)
          
          outputs.append(x)

        return outputs

    def tokenizer(self) -> RobertaTokenizer:
        return self._tokenizer