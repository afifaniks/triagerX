import json
import math
import os
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sentence_transformers import util

from triagerx.model.prediction_model import PredictionModel


class TriagerX:
    def __init__(
        self,
        component_prediction_model: PredictionModel,
        developer_prediction_model: PredictionModel,
        similarity_model: nn.Module,
        train_data: pd.DataFrame,
        train_embeddings: str,
        issues_path: str,
        developer_id_map: Dict[str, int],
        component_id_map: Dict[str, int],
        expected_developers: Set[str],
        device: str,
        similarity_prediction_weight: float,
        time_decay_factor: float,
        direct_assignment_score: float,
        contribution_score: float,
        discussion_score: float,
    ) -> None:
        self._component_prediction_model = component_prediction_model.to(device)
        self._developer_prediction_model = developer_prediction_model.to(device)
        self._similarity_model = similarity_model
        self._device = device
        self._similarity_prediction_weight = similarity_prediction_weight
        self._time_decay_factor = time_decay_factor
        self._direct_assignment_score = direct_assignment_score
        self._contribution_score = contribution_score
        self._discussion_score = discussion_score
        self._train_data = train_data
        self._issues_path = issues_path
        self._all_issues = os.listdir(issues_path)
        self._developer2id_map = developer_id_map
        self._component2id_map = component_id_map
        self._expected_developers = expected_developers
        self._id2developer_map = {idx: dev for dev, idx in developer_id_map.items()}
        self._id2component_map = {idx: comp for comp, idx in component_id_map.items()}
        self._all_embeddings = np.load(train_embeddings)
        logger.debug(f"Using device: {device}")
        logger.debug("Loading embeddings for existing issues...")

    def get_recommendation(
        self,
        issue: str,
        k_comp: int,
        k_dev: int,
        k_rank: int,
        similarity_threshold: float,
    ) -> Dict[str, List]:
        """
        Generates recommendations for components and developers based on the given issue.

        Args:
            issue (str): The issue for which recommendations are to be generated.
            k_comp (int): The number of top components to recommend.
            k_dev (int): The number of top developers to recommend.
            k_rank (int): The number of top ranked developers by similarity to consider.
            similarity_threshold (float): The threshold for developer similarity scores.

        Returns:
            Dict[str, List]: A dictionary containing recommended components, developers, and their scores.
        """
        predicted_components_name, comp_prediction_score = self._predict_components(
            issue, k_comp
        )

        logger.debug(f"Predicted components: {predicted_components_name}")
        logger.debug(f"Component prediction Score: {comp_prediction_score}")

        all_dev_prediction_scores = self._predict_developers(issue)

        topk_dev_prediction_score, topk_predicted_developers = torch.tensor(
            all_dev_prediction_scores
        ).topk(k_dev, dim=0, largest=True, sorted=True)
        topk_predicted_developers_name = [
            self._id2developer_map[idx]
            for idx in topk_predicted_developers.cpu().numpy()
        ]

        logger.debug(
            f"Predicted developers by the model: {topk_predicted_developers_name}"
        )
        logger.debug(
            f"Developer prediction score: {topk_dev_prediction_score.cpu().numpy().tolist()}"
        )

        (
            similarity_devs,
            normalized_similarity_score,
        ) = self._get_similarity_recommendations(issue, k_rank, similarity_threshold)

        logger.debug(
            f"Similar issue contribution scores: {list(zip(similarity_devs, normalized_similarity_score))}"
        )

        aggregated_rank, aggregated_prediction_score = self._aggregate_rankings(
            all_dev_prediction_scores,
            similarity_devs,
            normalized_similarity_score,
            k_dev,
        )

        logger.debug(f"Aggregated Ranks: {aggregated_rank}")
        logger.debug(f"Aggregated Score: {aggregated_prediction_score}")

        recommendations = {
            "predicted_components": predicted_components_name,
            "comp_prediction_score": comp_prediction_score,
            "predicted_developers": topk_predicted_developers_name,
            "dev_prediction_score": topk_dev_prediction_score,
            "similar_devs": similarity_devs[:k_dev],
            "similar_score": normalized_similarity_score[:k_dev],
            "combined_ranking": aggregated_rank,
            "combined_ranking_score": aggregated_prediction_score,
        }

        return recommendations

    def _predict_components(self, issue: str, k: int) -> Tuple[List[str], List[float]]:
        """
        Predicts components related to the given issue.

        Args:
            issue (str): The issue for which components are to be predicted.
            k (int): The number of top components to recommend.

        Returns:
            Tuple[List[str], List[float]]: A tuple containing a list of predicted component names and their prediction scores.
        """
        self._component_prediction_model.eval()
        with torch.no_grad():
            tokenized_issue = self._component_prediction_model.tokenize_text(issue)
            predictions = self._component_prediction_model(tokenized_issue)

        output = torch.sum(torch.stack(predictions), 0)
        output = self._normalize_tensor(output.squeeze(dim=0))
        prediction_score, predicted_components = output.topk(k, 0, True, True)
        predicted_components = (
            predicted_components.squeeze(dim=0).cpu().numpy().tolist()
        )

        predicted_components_name = [
            self._id2component_map[idx] for idx in predicted_components
        ]
        prediction_score = prediction_score.squeeze(dim=0).cpu().numpy().tolist()

        return predicted_components_name, prediction_score

    def _predict_developers(self, issue: str) -> np.ndarray:
        """
        Generates developer recommendations based on the given issue.

        Args:
            issue (str): The issue for which developers are to be recommended.

        Returns:
            np.ndarray: A numpy array of all developer's prediction scores.
        """
        self._developer_prediction_model.eval()
        with torch.no_grad():
            tokenized_issue = self._developer_prediction_model.tokenize_text(issue)
            predictions = self._developer_prediction_model(tokenized_issue)

        output = torch.sum(torch.stack(predictions), 0)
        normalized_score = self._normalize_tensor(output.squeeze(dim=0))

        return normalized_score.cpu().numpy()

    def _get_similarity_recommendations(
        self,
        issue: str,
        k_rank: int,
        similarity_threshold: float,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Generates recommendations based on developer similarity.

        Args:
            issue (str): The issue for which similarity recommendations are to be generated.
            predicted_components_name (List[str]): List of predicted component names.
            k_rank (int): The number of top ranked developers by similarity to consider.
            similarity_threshold (float): The threshold for developer similarity scores.

        Returns:
            Tuple[List[str], np.ndarray]: A tuple containing a list of similar developers and their normalized similarity scores.
        """
        similar_issues = self._get_top_k_similar_issues(
            issue, k_rank, similarity_threshold
        )
        dev_predictions_by_similarity = self._get_historical_contributors(
            similar_issues
        )

        similarity_devs = [dev[0] for dev in dev_predictions_by_similarity]
        similarity_scores = [score[1] for score in dev_predictions_by_similarity]

        normalized_similarity_score = np.array([])
        if similarity_scores:
            normalized_similarity_score = self._normalize(similarity_scores)

        return similarity_devs, normalized_similarity_score

    def _adjust_dev_scores_by_similarity(
        self,
        dev_prediction_score: np.ndarray,
        dev_predictions_by_similarity: List[str],
        normalized_similarity_score: np.ndarray,
    ) -> np.ndarray:
        """
        Adjusts developer scores based on similarity scores.

        Args:
            dev_prediction_score (np.ndarray): Normalized scores of all developers.
            dev_predictions_by_similarity (List[str]): List of developer predictions by similarity.
            normalized_similarity_score (np.ndarray): Normalized similarity scores.

        Returns:
            np.ndarray: An array of modified scores.
        """
        for i, sim_dev in enumerate(dev_predictions_by_similarity):
            dev_id = self._developer2id_map[sim_dev]
            dev_prediction_score[dev_id] += (
                normalized_similarity_score[i] * self._similarity_prediction_weight
            )

        return dev_prediction_score

    def _aggregate_rankings(
        self,
        dev_prediction_scores: np.ndarray,
        similarity_devs: List[str],
        normalized_similarity_score: np.ndarray,
        k_dev: int,
    ) -> Tuple[List[str], np.ndarray]:
        """
        Aggregates developer rankings.

        Args:
            dev_prediction_scores (List[float]): Prediction scores of all developers.
            normalized_similarity_score (np.ndarray): Normalized similarity scores.
            k_dev (int): The number of top developers to recommend.

        Returns:
            Tuple[List[str], np.ndarray]: A tuple containing the aggregated ranking of developers and the aggregated prediction score.
        """

        dev_prediction_scores = self._adjust_dev_scores_by_similarity(
            dev_prediction_scores, similarity_devs, normalized_similarity_score
        )
        prediction_scores_tensor = torch.tensor(dev_prediction_scores)
        (
            aggregated_prediction_score,
            aggregated_predicted_devs,
        ) = prediction_scores_tensor.topk(k_dev, 0, True, True)
        aggregated_rank = [
            self._id2developer_map[idx]
            for idx in aggregated_predicted_devs.cpu().numpy()
        ]

        return aggregated_rank, aggregated_prediction_score.cpu().numpy()

    def _get_historical_contributors(
        self, similar_issues: List[Tuple[int, float]]
    ) -> List[Tuple[str, float]]:
        """
        Retrieves historical contributors for similar issues.

        Args:
            similar_issues (List[Tuple[int, float]]): List of similar issues and their similarity scores.

        Returns:
            List[Tuple[str, float]]: A list of contributors and their contribution scores.
        """
        # Intialize all contribution score to 0
        user_contribution_scores = {dev: 0 for dev in self._developer2id_map.keys()}
        skipped_users = set()

        for issue_index, sim_score in similar_issues:
            issue = self._train_data.iloc[issue_index]
            issue_number = issue.issue_number
            contributors = self._get_contribution_data(issue_number)

            for key, users in contributors.items():
                for user_data in users:
                    user = user_data[0].lower()
                    created_at = user_data[1] if len(user_data) > 1 else None

                    if user not in self._expected_developers:
                        skipped_users.add(user)
                        continue

                    contribution_point = self._get_contribution_point(key)
                    time_decay = self._calculate_time_decay(created_at)

                    user_contribution_scores[user] = (
                        user_contribution_scores.get(user, 0)
                        + sim_score * contribution_point * time_decay
                    )

        if skipped_users:
            logger.warning(
                f"Skipped users: {skipped_users} because they don't exist in the expected developers list"
            )

        return sorted(
            user_contribution_scores.items(), key=lambda x: x[1], reverse=True
        )

    def _get_contribution_point(self, key: str) -> float:
        """
        Returns the contribution point for a given contribution type.

        Args:
            key (str): The contribution type key.

        Returns:
            float: The contribution point for the given type.
        """
        if key in ["pull_request", "commits"]:
            return self._contribution_score
        elif key in ["last_assignment", "direct_assignment"]:
            return self._direct_assignment_score
        return self._discussion_score

    def _calculate_time_decay(self, created_at: Optional[str]) -> float:
        """
        Calculates the time decay factor for a given creation time.

        Args:
            created_at (str): The creation time in ISO format.

        Returns:
            float: The time decay factor.
        """
        if not created_at:
            return 1

        contribution_date = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
        days_since_contribution = (datetime.now() - contribution_date).days
        return math.exp(-self._time_decay_factor * days_since_contribution)

    def _get_contribution_data(
        self, issue_number: int
    ) -> Dict[str, List[Tuple[str, str]]]:
        """
        Retrieves the contribution data for a given issue number.

        Args:
            issue_number (int): The issue number.

        Returns:
            Dict[str, List[Tuple[str, str]]]: A dictionary containing contribution data.
        """
        contributions = {}
        issue_file = f"{issue_number}.json"
        last_assignment = None

        if issue_file in self._all_issues:
            with open(os.path.join(self._issues_path, issue_file), "r") as file:
                issue = json.load(file)
                assignees = issue.get("assignees", [])
                assignee_logins = (
                    [(assignee["login"], None) for assignee in assignees]
                    if assignees
                    else []
                )
                contributions["direct_assignment"] = assignee_logins
                timeline = issue.get("timeline_data", [])
                pull_requests, commits, discussion = [], [], []

                for timeline_event in timeline:
                    event = timeline_event.get("event")
                    created_at = timeline_event.get("created_at")
                    actor = timeline_event.get("actor", {})

                    if not actor:
                        continue

                    actor = actor.get("login")

                    if event == "cross-referenced" and timeline_event["source"].get(
                        "issue", {}
                    ).get("pull_request"):
                        pull_requests.append((actor, created_at))
                        last_assignment = actor
                    elif event == "referenced" and timeline_event.get("commit_url"):
                        commits.append((actor, created_at))
                        last_assignment = actor
                    elif event == "commented":
                        discussion.append((actor, created_at))

                contributions["pull_request"] = pull_requests
                contributions["commits"] = commits
                contributions["discussion"] = discussion
                contributions["last_assignment"] = (
                    [(last_assignment, None)] if last_assignment else []
                )

        return contributions

    def _get_top_k_similar_issues(
        self, issue: str, k: int, threshold: float
    ) -> List[Tuple[int, float]]:
        """
        Retrieves the top k similar issues based on cosine similarity.

        Args:
            issue (str): The issue for which similar issues are to be found.
            k (int): The number of top similar issues to retrieve.
            threshold (float): The similarity threshold.

        Returns:
            List[Tuple[int, float]]: A list of issue indices and their similarity scores.
        """
        issue_embedding = self._similarity_model.encode(issue)
        cos_sim = util.cos_sim(issue_embedding, self._all_embeddings)
        topk_values, topk_indices = torch.topk(cos_sim, k=k)
        topk_values = topk_values.cpu().numpy()[0]
        topk_indices = topk_indices.cpu().numpy()[0]

        return [
            (idx, sim_score)
            for idx, sim_score in zip(topk_indices, topk_values)
            if sim_score >= threshold
        ]

    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalizes a numpy array of scores.

        Args:
            scores (np.ndarray): The array of scores to be normalized.

        Returns:
            np.ndarray: The normalized array of scores.
        """
        min_score = np.min(scores)
        max_score = np.max(scores)
        return (scores - min_score) / (max_score - min_score)

    def _normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalizes a tensor of scores.

        Args:
            tensor (torch.Tensor): The tensor of scores to be normalized.

        Returns:
            torch.Tensor: The normalized tensor of scores.
        """
        min_val = torch.min(tensor)
        max_val = torch.max(tensor)
        return (tensor - min_val) / (max_val - min_val)
