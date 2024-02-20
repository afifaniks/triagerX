import numpy as np
import torch
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from triagerx.dataset.triage_dataset import TriageDataset
from triagerx.trainer.train_config import TrainConfig


class ModelTrainer:
    def __init__(self, config: TrainConfig):
        self._config = config

    def _init_wandb(self):
        wandb.init(**self._config.wandb)

    def train(self, model: nn.Module):
        tokenizer = model.tokenizer()
        criterion = self._config.criterion
        optimizer = self._config.optimizer
        train_data = self._config.train_dataset
        validation_data = self._config.validation_dataset
        sampler = self._config.sampler

        train = TriageDataset(train_data, tokenizer)
        val = TriageDataset(validation_data, tokenizer)

        if self._config.wandb:
            logger.debug("Initializing wandb...")
            self._init_wandb()

        train_dataloader = DataLoader(
            dataset=train,
            batch_size=self._config.batch_size,
            shuffle=False if sampler else True,
            sampler=sampler,
        )
        val_dataloader = DataLoader(val, batch_size=self._config.batch_size)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_loss = float("inf")

        if torch.cuda.is_available():
            logger.debug(f"Selected compute device: {device}")
            model = model.cuda()
            criterion = criterion.cuda()

        for epoch_num in range(self._config.epochs):
            total_acc_train = 0
            total_loss_train = 0

            for train_input, train_label in tqdm(train_dataloader):
                train_label = train_label.to(device)
                mask = train_input["attention_mask"].to(device)
                input_id = train_input["input_ids"].squeeze(1).to(device)

                output = model(input_id, mask)

                batch_loss = criterion(output, train_label.long())
                total_loss_train += batch_loss.item()

                output = torch.sum(torch.stack(output), 0)

                acc = (output.argmax(dim=1) == train_label).sum().item()
                total_acc_train += acc

                model.zero_grad()
                batch_loss.backward()
                optimizer.step()

            total_acc_val = 0
            total_loss_val = 0
            correct_top_k = 0

            all_preds = []
            all_labels = []

            with torch.no_grad():

                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    mask = val_input["attention_mask"].to(device)
                    input_id = val_input["input_ids"].squeeze(1).to(device)

                    output = model(input_id, mask)

                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()

                    output = torch.sum(torch.stack(output), 0)
                    _, top_k_predictions = output.topk(10, 1, True, True)

                    top_k_predictions = top_k_predictions.t()

                    correct_top_k += (
                        top_k_predictions.eq(
                            val_label.view(1, -1).expand_as(top_k_predictions)
                        )
                        .sum()
                        .item()
                    )

                    acc = (output.argmax(dim=1) == val_label).sum().item()

                    all_preds.append(output.argmax(dim=1).cpu().numpy())
                    all_labels.append(val_label.cpu().numpy())

                    total_acc_val += acc

            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)

            precision, recall, f1_score, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="macro"
            )

            top10 = correct_top_k / len(validation_data)

            self._log_step(
                epoch_num,
                total_acc_train,
                total_acc_val,
                total_loss_train,
                total_loss_val,
                precision,
                recall,
                f1_score,
                train_data,
                validation_data,
                top10,
            )

            val_loss = total_loss_val / len(validation_data)

            if self._config.scheduler:
                self._config.scheduler.step(val_loss)

            if val_loss < best_loss:
                logger.success("Found new best model. Saving weights...")
                torch.save(model.state_dict(), self._config.output_file)
                best_loss = val_loss

        if self._config.wandb:
            wandb.finish()

    def _log_step(
        self,
        epoch_num,
        total_acc_train,
        total_acc_val,
        total_loss_train,
        total_loss_val,
        precision,
        recall,
        f1_score,
        train_data,
        validation_data,
        topk,
    ):
        log = f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} \
                    | Train Accuracy: {total_acc_train / len(train_data): .3f} \
                    | Val Loss: {total_loss_val / len(validation_data): .3f} \
                    | Val Accuracy: {total_acc_val / len(validation_data): .3f} \
                    | Top 10: {topk} \
                    | Precision: {precision: .3f} \
                    | Recall: {recall: .3f} \
                    | F1-score: {f1_score: .3f}"

        logger.info(log)

        if self._config.wandb:
            wandb.log(
                {
                    "train_acc": total_acc_train / len(train_data),
                    "train_loss": total_loss_train / len(train_data),
                    "val_acc": total_acc_val / len(validation_data),
                    "val_loss": total_loss_val / len(validation_data),
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1_score,
                    "top10": topk,
                }
            )
