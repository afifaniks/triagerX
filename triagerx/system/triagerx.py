import json
import os
import re
from typing import Any, Dict, List

import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sentence_transformers import SentenceTransformer, util
from transformers import PreTrainedTokenizer


class TriagerX:
    def __init__(
            self, 
            component_prediction_model: nn.Module, 
            developer_prediction_model: nn.Module,
            similarity_model: nn.Module,
            tokenizer: PreTrainedTokenizer,
            tokenizer_config: Dict,
            issues_path: str,
            train_data: pd.DataFrame,
            developer_id_map: Dict[str, int],
            component_id_map: Dict[str, int],
            expected_deveopers: List[str]
            ) -> None:
        self._component_prediction_model = component_prediction_model
        self._developer_prediction_model = developer_prediction_model
        self._similarity_model = similarity_model
        self._tokenizer = tokenizer
        self._tokenizer_config = tokenizer_config
        self._train_data = train_data
        self._special_tokens = {
            "hex": "[HEX]",
            "timestamp": "[TIMESTAMP]",
            "float": "[FLOAT_VALUE]",
            "param": "[PARAM_VALUE]"
        }
        self._issues_path = issues_path
        self._all_issues = os.listdir(self._issues_path)
        self._developer2id_map = developer_id_map
        self._component2id_map = component_id_map
        self._id2developer_map = {idx: dev for dev, idx in developer_id_map.items()}
        self._id2component_map = {idx: comp for comp, idx in component_id_map.items()}
        self._expected_deveopers = expected_deveopers
        logger.debug("Generating embedding for training data...")
        self._all_embeddings = similarity_model.encode(self._train_data.raw_text.to_list(), batch_size=15)

    def get_recommendation(self, issue, k=3):
        logger.debug("Processing issue...")
        cleaned_issue = self._clean_data(issue_data=issue)
        processed_issue = self._process_issues(issue=cleaned_issue)

        logger.debug("Prediciting components...")
        comp_prediction_score, predicted_components = self._get_predicted_components(tokenized_issue=processed_issue, k=k)
        print(predicted_components)
        predicted_components = predicted_components.squeeze(dim=0).cpu().numpy().tolist()
        predicted_components_name = [self._id2component_map[idx] for idx in predicted_components]
        logger.info(f"Predicted components: {predicted_components_name}")

        logger.debug("Generating developer recommendation...")
        dev_prediction_score, predicted_devs = self._get_recommendation_from_dev_model(tokenized_issue=processed_issue, k=5)
        predicted_devs = predicted_devs.squeeze(dim=0).cpu().numpy().tolist()
        dev_prediction_score = dev_prediction_score.squeeze(dim=0).cpu()
        predicted_developers_name = [self._id2developer_map[idx] for idx in predicted_devs]
        logger.info(f"Recommended developers: {predicted_developers_name}\nSoftmax: {dev_prediction_score}")

        dev_predictions_by_similarity = self._get_recommendation_by_similarity(
            issue, 
            predicted_components, 
            k_dev=5, 
            k_issue=10, 
            similarity_threshold=0.5
        )

        return predicted_devs, dev_predictions_by_similarity, predicted_components_name

    def _process_issues(self, issue: str):
        return self._tokenizer(
            issue,
            padding=self._tokenizer_config["padding"],
            max_length=self._tokenizer_config["max_length"],
            truncation=self._tokenizer_config["truncation"],
            return_tensors=self._tokenizer_config["return_tensors"]
        )

    def _get_recommendation_from_dev_model(self, tokenized_issue, k):
        with torch.no_grad():
            predictions = self._developer_prediction_model(
                input_ids=tokenized_issue["input_ids"],
                tok_type=tokenized_issue["token_type_ids"],
                attention_mask=tokenized_issue["attention_mask"]
            )

        output = torch.sum(torch.stack(predictions), 0)
        
        return output.topk(k, 1, True, True)
    
    def _get_predicted_components(self, tokenized_issue, k):
        with torch.no_grad():
            predictions = self._component_prediction_model(
                input_ids=tokenized_issue["input_ids"],
                tok_type=tokenized_issue["token_type_ids"],
                attention_mask=tokenized_issue["attention_mask"]
            )

        output = torch.sum(torch.stack(predictions), 0)
        return output.topk(k, 1, True, True)
    
    def _get_recommendation_by_similarity(self, issue, predicted_components, k_dev, k_issue, similarity_threshold):
        similar_issues = self._get_top_k_similar_issues(issue, k=k_issue, threshold=similarity_threshold)
        historical_contribution = self._get_historical_contributors(
            similar_issues=similar_issues, 
            predicted_component_ids=predicted_components
        )

        logger.debug(historical_contribution)

        top_k_devs = [contrib[0] for contrib in historical_contribution[:k_dev]]

        return top_k_devs

    def _get_historical_contributors(self, similar_issues, predicted_component_ids):
        user_contribution_counts = {}

        for issue_index, sim_score in similar_issues:
            base_points = sim_score
            
            issue = self._train_data.iloc[issue_index]
            # print("Issue Number", df_train.iloc[issue_index].issue_number, df_train.iloc[issue_index].issue_title)

            # print(f"Issue label: {label2idx[issue.component]} -- Predicted: {predicted_component_ids}")

            if self._component2id_map[issue.component] not in predicted_component_ids:
                logger.warning(f"Skipping issue as label id {self._component2id_map[issue.component]} did not match with any of {predicted_component_ids}")
                continue

            issue_number = issue.issue_number
            contributors = self._get_contribution_data(issue_number)

            for key, users in contributors.items():
                for user in users:
                    if user not in self._expected_deveopers:
                        logger.warning(f"Skipping: {user} because it doesn't exist in {{expected_developers}} list")
                        continue            
                                
                    # if user in components[issue.component]:
                    #     user_contribution_counts[user] = user_contribution_counts.get(user, 0) + base_points * 1.25
                    # else:
                    user_contribution_counts[user] = user_contribution_counts.get(user, 0) + base_points
            
            # base_points /= 2.0
        
        user_contribution_counts = sorted(user_contribution_counts.items(), key=lambda x: x[1], reverse=True)
        return user_contribution_counts

    def _get_contribution_data(self, issue_number):
        contributions = {}
        issue_file = f"{issue_number}.json"
        last_assignment = None
        
        if issue_file in self._all_issues:
            with open(os.path.join(self._issues_path, issue_file), "r") as file:
                issue = json.load(file)      

                assignees = issue["assignees"]
                assignee_logins = (
                    [assignee["login"] for assignee in assignees] if len(assignees) > 0 else []
                )

                contributions["direct_assignment"] = assignee_logins

                timeline = issue["timeline_data"]
                pull_requests = []
                commits = []
                discussion = []

                for timeline_event in timeline:
                    event = timeline_event["event"]

                    if event == "cross-referenced" and timeline_event["source"]["issue"].get("pull_request", None):
                        actor = timeline_event["actor"]["login"]
                        pull_requests.append(actor)
                        last_assignment = actor

                    if event == "referenced" and timeline_event["commit_url"]:
                        actor = timeline_event["actor"]["login"]
                        commits.append(actor)
                        last_assignment = actor

                    if event == "commented":
                        actor = timeline_event["actor"]["login"]
                        discussion.append(actor)                    
                
                contributions["direct_assignment"] = assignee_logins
                contributions["pull_request"] = pull_requests
                contributions["commits"] = commits
                contributions["discussion"] = discussion
                contributions["last_assignment"] = [last_assignment] if last_assignment else []

        return contributions
    
    def _get_top_k_similar_issues(self, issue, k, threshold):
        test_embed = self._similarity_model.encode(issue)
        cos = util.cos_sim(test_embed, self._all_embeddings)
        
        topk_values, topk_indices = torch.topk(cos, k=k)
        topk_values = topk_values.cpu().numpy()[0]
        topk_indices = topk_indices.cpu().numpy()[0]
        
        similar_issues = []
        
        for idx, sim_score in zip(topk_indices, topk_values):
            if sim_score >= threshold:
                similar_issues.append([idx, sim_score])

        return similar_issues
    
    def _clean_data(self, issue_data):
        issue_data = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', issue_data)
        issue_data = re.sub(" +", " ", issue_data)
        issue_data = issue_data.strip()
        issue_data = re.sub(r'0x[\da-fA-F]+', self._special_tokens["hex"], issue_data)
        issue_data = re.sub(r'\b[0-9a-fA-F]{16}\b', self._special_tokens["hex"], issue_data)
        issue_data = re.sub(r'\b\d{2}:\d{2}:\d{2}\.\d{3}\b', self._special_tokens["timestamp"], issue_data)
        issue_data = re.sub(r'\s*[-+]?\d*\.\d+([eE][-+]?\d+)?', self._special_tokens["float"], issue_data)
        issue_data = re.sub(r'=\s*-?\d+', f'= {self._special_tokens["param"]}', issue_data)

        return issue_data