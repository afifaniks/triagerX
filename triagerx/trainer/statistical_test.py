import json
from typing import List

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from scipy.stats import ttest_rel
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm


class StatisticalEvaluator:
    def evaluate(
        self,
        model: nn.Module,
        model2: nn.Module,
        device: str,
        dataloader: DataLoader,
        dataloader2: DataLoader,
        topk_indices: List[int],
        combined_loss: bool = True,
    ):
        model = model.to(device)
        model2 = model2.to(device)
        model.eval()
        model2.eval()

        all_preds = []
        all_preds2 = []
        all_labels = []
        all_logits1 = []
        all_logits2 = []

        index = 0

        with torch.no_grad():
            for (test_input, test_label), (test_input2, test_label2) in tqdm(
                zip(dataloader, dataloader2), desc="Test Steps"
            ):
                test_label = test_label.to(device)
                test_label2 = test_label2.to(device)

                assert np.array_equal(
                    test_label.cpu().numpy(), test_label2.cpu().numpy()
                )

                output = model(test_input)
                output2 = model2(test_input2)

                if combined_loss:
                    output = torch.sum(torch.stack(output), 0)
                    output2 = torch.sum(torch.stack(output2), 0)

                # _, top_k_predictions = output.topk(max(topk_indices), 1, True, True)
                # _, top_k_predictions2 = output2.topk(max(topk_indices), 1, True, True)

                # top_k_predictions = top_k_predictions.t()
                # top_k_predictions2 = top_k_predictions2.t()

                all_preds.append(output.argmax(dim=1).cpu().numpy())
                all_preds2.append(output2.argmax(dim=1).cpu().numpy())
                all_labels.append(test_label.cpu().numpy())

                all_logits1.append(output.cpu().numpy())
                all_logits2.append(output2.cpu().numpy())

                index += 1
                if index > 6:
                    break

        # Concatenate full arrays for p-value test
        logits_model1 = np.concatenate(all_logits1, axis=0)
        logits_model2 = np.concatenate(all_logits2, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)

        p_values = self.p_value_test(
            labels_np, logits_model1, logits_model2, topk_indices, subset_size=50
        )

        logger.info(f"P-values from top-k comparison: {p_values}")

    def p_value_test(
        self,
        labels,
        model1_preds,
        model2_preds,
        topk_indices,
        num_samples=50,
        subset_size=None,
    ):
        """
        Compute p-values comparing top-k accuracies of two models over multiple random subsets.
        """
        labels = np.array(labels)
        model1_preds = np.array(model1_preds)
        model2_preds = np.array(model2_preds)

        if subset_size is None:
            subset_size = len(labels)

        p_values = {}

        def compute_topk_accuracy(logits, labels, k):
            topk = np.argsort(logits, axis=1)[:, -k:]
            return np.mean([label in topk_row for label, topk_row in zip(labels, topk)])

        for k in topk_indices:
            accs_model1 = []
            accs_model2 = []

            for _ in range(num_samples):
                subset_indices = np.random.choice(
                    len(labels), size=subset_size, replace=False
                )
                subset_labels = labels[subset_indices]
                subset_preds1 = model1_preds[subset_indices]
                subset_preds2 = model2_preds[subset_indices]

                acc1 = compute_topk_accuracy(subset_preds1, subset_labels, k)
                acc2 = compute_topk_accuracy(subset_preds2, subset_labels, k)

                accs_model1.append(acc1)
                accs_model2.append(acc2)

            t_stat, p_val = ttest_rel(accs_model1, accs_model2)
            p_values[k] = p_val

        return p_values

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
