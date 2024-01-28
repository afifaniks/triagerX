import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer


class RobertaFCNClassifier(nn.Module):
    def __init__(
        self, 
        model_name: str, 
        output_size, 
        embed_size=1024, 
        dropout=0.1
    ) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(model_name)

        for p in self.base.embeddings.parameters():
                p.requires_grad = False

        for i in range (0, 20):
            for p in self.base.encoder.layer[i].parameters():
                p.requires_grad = False

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.linear = nn.Linear(embed_size, embed_size // 2)
        self.linear2 = nn.Linear(embed_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, input_id, attention_mask):
        _, pooler_out = self.base(
            input_ids=input_id, attention_mask=attention_mask, return_dict=False
        )
        pooler_out = self.relu(pooler_out)
        drop_out = self.dropout(pooler_out)
        linear_out = self.linear(drop_out)
        linear_out = self.relu(linear_out)
        drop_out = self.dropout2(linear_out)
        linear_out = self.linear2(drop_out)

        return linear_out

    def tokenizer(self) -> PreTrainedTokenizer:
        return self._tokenizer