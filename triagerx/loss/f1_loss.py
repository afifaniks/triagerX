import torch
import torch.nn as nn
import torch.nn.functional as F


class F1Loss(nn.Module):
    def __init__(self):
        super(F1Loss, self).__init__()

    def forward(self, predictions, y_true):
        loss = 0

        for i in range(len(predictions)):
            y_pred = predictions[i]
            
            y_probs = F.softmax(y_pred, dim=1)
        
            # Convert true labels to one-hot encoding
            y_onehot = F.one_hot(y_true, num_classes=y_probs.size(1)).float()

            # Calculate true positives, false positives, and false negatives
            true_positives = (y_probs * y_onehot).sum(dim=0)
            false_positives = ((1 - y_onehot) * y_probs).sum(dim=0)
            false_negatives = (y_onehot * (1 - y_probs)).sum(dim=0)

            # Calculate precision and recall
            precision = true_positives / (true_positives + false_positives + 1e-10)
            recall = true_positives / (true_positives + false_negatives + 1e-10)

            # Calculate F1 score for each class
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)

            # Average F1 scores across all classes
            mean_f1_score = f1_score.mean()

            loss += (1 - mean_f1_score)         

        return loss