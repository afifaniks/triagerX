import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer


class CodeRobertaClassifier(nn.Module):
    def __init__(
        self,
        output_size,
        text_model: str = "roberta-base",
        code_model: str = "microsoft/codebert-base",
        embed_size=768,
        dropout=0.1
    ) -> None:
        super().__init__()
        self.text_model = AutoModel.from_pretrained(
            text_model
        )
        self.code_model = AutoModel.from_pretrained(
            code_model
        )
        self._text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self._code_tokenizer = AutoTokenizer.from_pretrained(code_model)
        self.linear = nn.Linear(embed_size*2, output_size)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, text_input_ids, text_mask, code_input_ids, code_mask):
        _, text_pooler_out = self.text_model(text_input_ids, text_mask, return_dict=False)
        _, code_pooler_out = self.code_model(code_input_ids, code_mask, return_dict=False)
        out_cat = torch.cat([text_pooler_out, code_pooler_out], dim=1)
        out_cat = self.dropout(out_cat)

        return self.relu(self.linear(out_cat))
