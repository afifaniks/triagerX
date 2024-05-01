from typing import Any, Dict, List

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer


class TriagerX:
    def __init__(
            self, 
            component_prediction_model: nn.Module, 
            developer_prediction_model: nn.Module,
            similarity_model: nn.Module,
            tokenizer: PreTrainedTokenizer,
            tokenizer_config: Dict,
            issues_path: str
            ) -> None:
        self._component_prediction_model = component_prediction_model
        self._developer_prediction_model = developer_prediction_model
        self._similarity_model = similarity_model
        self._tokenizer = tokenizer
        self._tokenizer_config = tokenizer_config

    def get_recommendation(self, issue, k=3):
        processed_issue = self._process_issues(issue=issue)
        dev_predictions = self._get_recommendation_from_dev_model(tokenized_issue=processed_issue, k=k)

        return dev_predictions

    def _process_issues(self, issue: str):
        return self._tokenizer(
            issue,
            padding=self._tokenizer_config["padding"],
            max_length=self._tokenizer_config["max_length"],
            truncation=self._tokenizer_config["truncation"],
            return_tensors=self._tokenizer_config["return_tensors"]
        )

    def _get_recommendation_from_dev_model(self, tokenized_issue, k):
        predictions = self._developer_prediction_model(
            input_ids=tokenized_issue["input_ids"],
            tok_type=tokenized_issue["token_type_ids"],
            attention_mask=tokenized_issue["attention_mask"]
        )

        output = torch.sum(torch.stack(predictions), 0)
        return output.topk(k, 1, True, True)