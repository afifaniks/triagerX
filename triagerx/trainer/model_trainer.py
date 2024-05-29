import numpy as np
import torch
from loguru import logger
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from triagerx.trainer.train_config import TrainConfig
from triagerx.utils.early_stopping import EarlyStopping


class ModelTrainer:
    def __init__(self, config: TrainConfig):
        self._config = config

    def train(self):
        model = self._config.model.to(self._config.device)
        criterion = self._config.criterion.to(self._config.device)
        optimizer = self._config.optimizer
        scheduler = self._config.scheduler
        early_stopping = (
            EarlyStopping(patience=self._config.early_stopping_patience)
            if self._config.early_stopping_patience
            else None
        )
        train_dataloader = self._config.train_dataloader
        validation_dataloader = self._config.validation_dataloader
        best_loss = float("inf")

        for epoch_num in range(self._config.epochs):
            total_acc_train, total_loss_train = self._train_one_epoch(
                model, train_dataloader, criterion, optimizer, scheduler
            )
            (
                total_acc_val,
                total_loss_val,
                precision,
                recall,
                f1_score,
                topk,
            ) = self._validate_one_epoch(model, validation_dataloader, criterion)

            log_metrics = {
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "val_loss": total_loss_val,
                "val_acc": total_acc_val,
                f"top{self._config.topk_indices}_acc": topk,
                "train_loss": total_loss_train,
                "train_acc": total_acc_train,
            }
            self._config.log_manager.log_epoch(
                epoch_num=epoch_num,
                total_epochs=self._config.epochs,
                metrics=log_metrics,
            )

            if early_stopping:
                early_stopping(val_loss=total_loss_val)
                if early_stopping.early_stop:
                    logger.debug(
                        f"Validation loss did not improve for {early_stopping.patience} epochs. Early stopping..."
                    )
                    break

            if total_loss_val < best_loss:
                best_loss = total_loss_val
                logger.success(
                    f"Validation loss decreased, saving chekpoint to {self._config.output_path}..."
                )
                self.save_checkpoint(model, self._config.output_path)

    def _train_one_epoch(self, model, dataloader, criterion, optimizer, scheduler):
        total_acc_train = 0
        total_loss_train = 0
        model.train()

        for train_input, train_label in tqdm(dataloader, desc="Training Steps"):
            optimizer.zero_grad()
            train_label = train_label.to(self._config.device)
            mask = train_input["attention_mask"].squeeze(1).to(self._config.device)
            input_id = train_input["input_ids"].squeeze(1).to(self._config.device)
            tok_type = train_input["token_type_ids"].squeeze(1).to(self._config.device)
            output = model(input_id, mask, tok_type)

            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            output = torch.sum(torch.stack(output), 0)
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            batch_loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

        return total_acc_train / len(dataloader.dataset), total_loss_train / len(
            dataloader.dataset
        )

    def _validate_one_epoch(self, model, dataloader, criterion):
        total_acc_val = 0
        total_loss_val = 0
        correct_top_k = 0
        all_preds = []
        all_labels = []

        model.eval()
        with torch.no_grad():
            for val_input, val_label in tqdm(dataloader, desc="Validation Steps"):
                val_label = val_label.to(self._config.device)
                mask = val_input["attention_mask"].squeeze(1).to(self._config.device)
                input_id = val_input["input_ids"].squeeze(1).to(self._config.device)
                tok_type = (
                    val_input["token_type_ids"].squeeze(1).to(self._config.device)
                )
                output = model(input_id, mask, tok_type)

                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                output = torch.sum(torch.stack(output), 0)
                _, top_k_predictions = output.topk(
                    self._config.topk_indices, 1, True, True
                )
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
        topk = correct_top_k / len(dataloader.dataset)
        total_loss_val = total_loss_val / len(dataloader.dataset)
        total_acc_val = total_acc_val / len(dataloader.dataset)

        return total_acc_val, total_loss_val, precision, recall, f1_score, topk

    def save_checkpoint(self, model, output_path):
        torch.save(model.state_dict(), output_path)
