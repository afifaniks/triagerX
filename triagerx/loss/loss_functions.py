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

            loss += 1 - mean_f1_score

        return loss


class F1CELoss(nn.Module):
    def __init__(self, num_classes, ce_weigths=None, beta=1):
        super(F1CELoss, self).__init__()
        self._num_classes = num_classes
        self._ce = nn.CrossEntropyLoss(weight=ce_weigths)
        self._beta = beta

    def forward(self, predictions, y_true):
        total_loss = 0

        for i in range(len(predictions)):
            y_pred = predictions[i]

            ce_loss = self._ce(y_pred, y_true)

            y_probs = F.softmax(y_pred, dim=1)

            # Convert true labels to one-hot encoding
            y_onehot = F.one_hot(y_true, num_classes=self._num_classes).float()

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

            # Make sure losses are in scale of 0...1
            loss = (self._beta * ce_loss) + ((1 - mean_f1_score) * (1 - self._beta))
            total_loss += loss

        return total_loss  # You can take a mean


class CombinedLoss(nn.Module):
    def __init__(self, weights=None) -> None:
        super().__init__()
        self._ce = nn.CrossEntropyLoss(weight=weights)

    def forward(self, prediction, labels) -> torch.Tensor:
        loss = 0

        for i in range(len(prediction)):
            loss += self._ce(prediction[i], labels)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            focal_loss = focal_loss * self.alpha

        if self.reduction == "mean":
            return torch.mean(focal_loss)
        elif self.reduction == "sum":
            return torch.sum(focal_loss)
        else:
            return focal_loss


class CombinedFocalLoss(nn.Module):
    def __init__(self, reduction="mean") -> None:
        super().__init__()
        self._fl = FocalLoss(reduction=reduction)

    def forward(self, prediction, labels) -> torch.Tensor:
        loss = 0

        for i in range(len(prediction)):
            loss += self._fl(prediction[i], labels)

        return loss


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # Normalize the feature vectors
        features = F.normalize(features, dim=1)

        # Compute the similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create masks for positive and negative pairs
        labels = labels.unsqueeze(1)
        mask_positive = torch.eq(labels, labels.T).float()
        mask_negative = 1 - mask_positive

        # Compute log-softmax of the similarity matrix
        log_prob = F.log_softmax(similarity_matrix, dim=1)

        # Compute the positive log-probabilities
        positive_log_prob = (log_prob * mask_positive).sum(1) / mask_positive.sum(1)

        # Compute the negative log-probabilities
        negative_log_prob = (log_prob * mask_negative).sum(1) / mask_negative.sum(1)

        # Combine positive and negative log-probabilities to get the contrastive loss
        contrastive_loss = -(positive_log_prob - negative_log_prob).mean()

        return contrastive_loss
