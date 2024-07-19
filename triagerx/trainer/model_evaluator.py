import json
from typing import List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm


class ModelEvaluator:
    def evaluate(
        self,
        model: nn.Module,
        device: str,
        dataloader: DataLoader,
        run_name: str,
        topk_indices: List[int],
        weights_save_location: str,
        test_report_location: str,
        combined_loss: bool = True,
    ):
        model = model.to(device)
        model.eval()
        correct_top_k = {k: 0 for k in topk_indices}
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for test_input, test_label in tqdm(dataloader, desc="Test Steps"):
                test_label = test_label.to(device)
                output = model(test_input)

                if combined_loss:
                    output = torch.sum(torch.stack(output), 0)

                _, top_k_predictions = output.topk(max(topk_indices), 1, True, True)
                top_k_predictions = top_k_predictions.t()
                for k in topk_indices:
                    correct_top_k[k] += self._count_correct_predictions(
                        top_k_predictions[:k], test_label
                    )

                all_preds.append(output.argmax(dim=1).cpu().numpy())
                all_labels.append(test_label.cpu().numpy())

        accuracy_top_k = {
            k: correct_top_k[k] / len(dataloader.dataset) for k in topk_indices
        }

        logger.info(f"Correct top k prediction: {accuracy_top_k}")

        all_preds_np = np.concatenate(all_preds)
        all_labels_np = np.concatenate(all_labels)

        report = self.generate_classification_report(
            all_labels_np,
            all_preds_np,
            accuracy_top_k,
            run_name,
            weights_save_location,
        )
        self.save_report(report, test_report_location)
        logger.info(f"Classification Report:\n{report}")
        logger.info(f"Classification report saved at: {test_report_location}")

    def generate_classification_report(
        self,
        all_labels,
        all_preds,
        topk_scores,
        run_name,
        weights_save_location,
    ):
        report = classification_report(all_labels, all_preds, output_dict=True)

        for k_index, k_score in topk_scores.items():
            report[f"top{k_index}_acc"] = k_score

        report["run_name"] = run_name
        report["model_location"] = weights_save_location

        return report

    @staticmethod
    def save_report(report, test_report_location):
        with open(test_report_location, "w") as output_file:
            json.dump(report, output_file, indent=2)

    @staticmethod
    def _count_correct_predictions(top_k_predictions, test_label):
        """Count correct predictions for a given top-k setting."""
        return (
            top_k_predictions.eq(test_label.view(1, -1).expand_as(top_k_predictions))
            .sum()
            .item()
        )
