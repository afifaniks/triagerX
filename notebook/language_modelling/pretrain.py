import math
import re

import pandas as pd
import torch
from accelerate import Accelerator
from datasets import Dataset as HfDataset
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling,
                          DebertaConfig, DebertaForMaskedLM, RobertaConfig,
                          RobertaForMaskedLM, Trainer, TrainerCallback,
                          TrainingArguments)

import wandb


class TextProcessor:
    def __init__(self, tokenizer, train_file):
        self.tokenizer = tokenizer
        self.train_file = train_file
        self.df = pd.read_csv(self.train_file)
        self.texts = self._preprocess_text()

    def _preprocess_text(self):
        texts = [self._clean_text(f"Issue Title: {row.issue_title} Issue Description: {row.description}") for _, row in tqdm(self.df.iterrows(), desc="Preprocessing text")]
        self.df["text"] = texts
        return texts

    def _clean_text(self, line):
        line = re.sub(r'-+', ' ', line)
        line = re.sub(r'[^a-zA-Z, ]+', " ", line)
        line = re.sub(r'[ ]+', " ", line)
        line += "."
        return line
    
    def train_tokenizer(self, vocab_size=20000, save_path="tokenizer/deeptriage"):
        training_corpus = (self.texts[i: i + 1000] for i in range(0, len(self.texts), 1000))
        tokenizer_trained = self.tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        tokenizer_trained.save_pretrained(save_path)


    def tokenize_texts(self, max_length):
        tokenized_texts = self.tokenizer(self.texts, truncation=True, padding=True)
        return tokenized_texts

class CustomDataset(Dataset):
    def __init__(self, tokenizer, raw_datasets, max_length):
        self.tokenizer = tokenizer
        self.raw_datasets = raw_datasets
        self.max_length = max_length
        self.tokenized_datasets = self._tokenize_dataset()

    def _tokenize_dataset(self):
        accelerator = Accelerator(gradient_accumulation_steps=1)

        with accelerator.main_process_first():
            tokenized_datasets = self.raw_datasets.map(
                self._tokenize_function,
                batched=True,
                num_proc=4,
                remove_columns=['text'],
                desc="Running tokenizer on dataset line_by_line"
            )
            tokenized_datasets.set_format('torch', columns=['input_ids'], dtype=torch.long)
        return tokenized_datasets

    def _tokenize_function(self, examples):
        examples['text'] = [line for line in examples['text'] if len(line[0]) > 0 and not line[0].isspace()]
        return self.tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,            
            max_length=self.max_length,
            return_special_tokens_mask=True
        )
    def __len__(self):
        return len(self.tokenized_datasets)

    def __getitem__(self, i):
        return self.tokenized_datasets[i]


def train_model(model, args, data_collator, train_dataset, eval_dataset):
    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    return trainer

def evaluate_model(trainer):
    results = trainer.evaluate()
    perplexity = math.exp(results['eval_loss'])
    print(f">>> Perplexity: {perplexity:.2f}")

def save_text_to_file(texts, file_path):
    with open(file_path, "w") as f:
        for text in tqdm(texts):
            f.write(text + "\n")

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("roberta-large")
    train_file = "/home/mdafifal.mamun/notebooks/triagerX/notebook/data/deeptriage/deep_data.csv"
    
    # Preprocess and tokenize texts
    logger.info("Processing texts...")
    text_processor = TextProcessor(tokenizer, train_file)

    # Prepare datasets
    X_train, y_train = train_test_split(text_processor.df, test_size=0.2, shuffle=True)

    logger.info("Preparing datasets...")
    train_dataset = HfDataset.from_pandas(X_train)
    test_dataset = HfDataset.from_pandas(y_train)
    train_line_dataset = CustomDataset(tokenizer, train_dataset, 512)
    test_line_dataset = CustomDataset(tokenizer, test_dataset, 512)

    logger.info("Loading base model...")
    config = RobertaConfig.from_pretrained("roberta-large")
    config.num_hidden_layers = 6

    model = RobertaForMaskedLM(config)

    wandb.init(
        project="deberta-pretraining",
        name=f"roberta_{config.num_hidden_layers}_1024_10000steps"
    )
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./roberta-mlm-full",
        overwrite_output_dir=True,
        learning_rate=1e-5,
        per_device_train_batch_size=15,
        per_device_eval_batch_size=15,
        max_steps=10000,
        eval_steps=100,
        weight_decay=0.01,
        evaluation_strategy="steps",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="steps",
        report_to="wandb",
        load_best_model_at_end=True,
    )

    # Train model
    # config = DebertaConfig(
    #     hidden_size=1020,
    #     vocab_size=tokenizer.vocab_size,
    #     num_hidden_layers=6,
    #     num_attention_heads=12,
    #     intermediate_size=1024,
    #     max_position_embeddings=512
    # )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    logger.info("Training...")
    trainer = train_model(model, training_args, data_collator, train_line_dataset, test_line_dataset)
    wandb.finish()

    # Evaluate model
    evaluate_model(trainer)

    # Save texts to file
    save_text_to_file(text_processor.texts, "mlm_training_data.txt")
