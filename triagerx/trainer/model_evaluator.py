import json

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
        topk_index: int,
        weights_save_location: str,
        test_report_location: str,
    ):
        model.eval()
        correct_top_k_wo_sim = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for test_input, test_label in tqdm(dataloader, desc="Test Steps"):
                test_label = test_label.to(device)
                mask = test_input["attention_mask"].squeeze(1).to(device)
                input_id = test_input["input_ids"].squeeze(1).to(device)
                tok_type = test_input["token_type_ids"].squeeze(1).to(device)

                output = model(input_id, mask, tok_type)
                output = torch.sum(torch.stack(output), 0)

                _, top_k_wo_sim = output.topk(topk_index, 1, True, True)
                top_k_wo_sim = top_k_wo_sim.t()
                correct_top_k_wo_sim += (
                    top_k_wo_sim.eq(test_label.view(1, -1).expand_as(top_k_wo_sim))
                    .sum()
                    .item()
                )

                all_preds.append(output.argmax(dim=1).cpu().numpy())
                all_labels.append(test_label.cpu().numpy())

        logger.info(
            f"Correct top {topk_index} prediction: {correct_top_k_wo_sim}, {correct_top_k_wo_sim / len(dataloader.dataset)}"
        )

        all_preds_np = np.concatenate(all_preds)
        all_labels_np = np.concatenate(all_labels)

        report = self.generate_classification_report(
            all_labels_np,
            all_preds_np,
            correct_top_k_wo_sim,
            len(dataloader.dataset),
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
        correct_top_k_wo_sim,
        num_samples,
        run_name,
        weights_save_location,
    ):
        report = classification_report(all_labels, all_preds, output_dict=True)

        report["test_accuracy"] = correct_top_k_wo_sim / num_samples
        report["run_name"] = run_name
        report["model_location"] = weights_save_location

        return report

    @staticmethod
    def save_report(report, test_report_location):
        with open(test_report_location, "w") as output_file:
            json.dump(report, output_file)
