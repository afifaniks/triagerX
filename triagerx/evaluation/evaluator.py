import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import top_k_accuracy_score


class Evaluator:
    def calculate_top_k_accuray(
        self, model: nn.Module, k: int, X_test: pd.DataFrame, y_test: np.array
    ):
        cuda = torch.cuda.is_available()

        if cuda:
            model = model.cuda()

        tokenizer = model.tokenizer()

        y_preds = []

        logger.debug("Calculating predications...")
        for i in range(len(X_test)):
            dx = X_test.iloc[i]

            data = tokenizer(
                dx["text"], padding="max_length", max_length=512, truncation=True
            )
            ids, mask = data["input_ids"], data["attention_mask"]

            ids = torch.tensor([ids])
            mask = torch.tensor([mask])

            if cuda:
                ids = ids.cuda()
                mask = mask.cuda()

            softmax = nn.Softmax(dim=1)

            with torch.no_grad():
                # y_pred = softmax(model(ids, mask))
                y_pred = softmax(torch.sum(torch.stack(model(ids, mask)), 0))

            y_preds.append(y_pred)

        y_numpy = []

        for y in y_preds:
            y_numpy.append(y.cpu().numpy())

        y_preds = np.array(y_numpy)[:, 0, :]

        logger.debug(f"Calculating top {k} score...")

        score = top_k_accuracy_score(y_test, y_preds, k=k)
        logger.info(f"Top {k} score: {score}")

        return score
