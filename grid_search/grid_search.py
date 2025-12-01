import os
import csv
import torch
import numpy as np
from tqdm import tqdm
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification, ElectraConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from itertools import product
from utils import *

TRAIT = "Openness"
class CrossValidationTrainer:
    def __init__(self, lr, batch_size, weight_decay, warmup_ratio):
        self.csv_path = "./dataset/augmentation/contextual_augmentation/augmented_contextual.csv"
        self.trait_name = TRAIT
        self.k_folds = 5

        self.model_name = "google/electra-base-discriminator"
        self.tokenizer = ElectraTokenizerFast.from_pretrained(self.model_name)

        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.epochs = 5

        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def create_model(self, num_training_steps):
        config = ElectraConfig.from_pretrained(
            self.model_name,
            num_labels=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

        model = ElectraForSequenceClassification.from_pretrained(self.model_name, config=config).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        warmup_steps = int(num_training_steps * self.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)

        return model, optimizer, scheduler

    def evaluate(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = torch.argmax(outputs.logits, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0)
        }

    def compute_train_metrics(self, preds, labels):
        return {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0)
        }

    def train_fold(self, train_df, val_df, fold):
        train_ds = EssayDataset(train_df, self.tokenizer)
        val_ds = EssayDataset(val_df, self.tokenizer)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        total_steps = len(train_loader) * self.epochs
        model, optimizer, scheduler = self.create_model(total_steps)

        epoch_logs = []

        for epoch in range(1, self.epochs + 1):
            model.train()
            total_loss = 0
            preds_all, labels_all = [], []

            loop = tqdm(train_loader, desc=f"Training Fold {fold+1} Epoch {epoch}")

            for batch in loop:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                preds_all.extend(preds.cpu().tolist())
                labels_all.extend(batch["labels"].cpu().tolist())

            train_loss = total_loss / len(train_loader)
            train_metrics = self.compute_train_metrics(preds_all, labels_all)

            val_loss = 0
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
            val_loss /= len(val_loader)

            val_metrics = self.evaluate(model, val_loader)

            # Save epoch log
            epoch_logs.append({
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                "val_loss": val_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()}
            })

        return epoch_logs

    def run(self):
        df = load_split_dataset(self.csv_path, trait_name=self.trait_name, split=False)
        kf = KFolder(n_splits=self.k_folds, shuffle=True, random_state=42)

        all_folds = []

        for train_df, val_df, fold_idx in kf.kfold_split(df):
            fold_log = self.train_fold(train_df, val_df, fold_idx)
            all_folds.append(fold_log)

        return all_folds

class ResultLogger:
    def __init__(self, save_dir="grid_search_results"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.txt_path = os.path.join(save_dir, f"{TRAIT}_gridsearch_results.txt")
        self.csv_path = os.path.join(save_dir, f"{TRAIT}_gridsearch_results.csv")

        # Create CSV header if new file
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "config_id", "lr", "batch_size", "weight_decay", "warmup_ratio",
                    "fold", "epoch",
                    "train_loss", "train_precision", "train_recall", "train_f1", "train_accuracy",
                    "val_loss", "val_precision", "val_recall", "val_f1", "val_accuracy"
                ])

    def log_txt(self, text):
        with open(self.txt_path, "a") as f:
            f.write(text + "\n")

    def log_csv(self, config_id, params, fold, epoch_data):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                config_id,
                params["lr"], params["batch_size"], params["weight_decay"], params["warmup_ratio"],
                fold,
                epoch_data["epoch"],
                epoch_data["train_loss"],
                epoch_data["train_precision"],
                epoch_data["train_recall"],
                epoch_data["train_f1"],
                epoch_data["train_accuracy"],
                epoch_data["val_loss"],
                epoch_data["val_precision"],
                epoch_data["val_recall"],
                epoch_data["val_f1"],
                epoch_data["val_accuracy"],
            ])

def generate_param_combinations(param_grid):
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in product(*values):
        yield dict(zip(keys, combo))

def run_grid_search():
    logger = ResultLogger()
    config_idx = 1

    for params in generate_param_combinations(hyperparameter_grid):
        header = "="*60
        print(f"\n{header}\n=== CONFIG {config_idx} ===\n{params}\n{header}\n")

        logger.log_txt(f"{header}\nCONFIG {config_idx}\n{params}\n{header}")

        trainer = CrossValidationTrainer(**params)
        folds = trainer.run()

        for fold_id, fold in enumerate(folds, 1):

            logger.log_txt(f"\n--- Fold {fold_id} ---")

            for ep in fold:
                # --- PRINT ---
                print(f"\nEpoch {ep['epoch']}")
                print(f"  Train Loss:      {ep['train_loss']:.4f}")
                print(f"  Train Precision: {ep['train_precision']:.4f}")
                print(f"  Train Recall:    {ep['train_recall']:.4f}")
                print(f"  Train F1:        {ep['train_f1']:.4f}")
                print(f"  Train Accuracy:  {ep['train_accuracy']:.4f}")

                print(f"  Val Loss:        {ep['val_loss']:.4f}")
                print(f"  Val Precision:   {ep['val_precision']:.4f}")
                print(f"  Val Recall:      {ep['val_recall']:.4f}")
                print(f"  Val F1:          {ep['val_f1']:.4f}")
                print(f"  Val Accuracy:    {ep['val_accuracy']:.4f}")

                # --- SAVE TO TXT ---
                logger.log_txt(
                    f"\nEpoch {ep['epoch']}\n"
                    f"  Train Loss: {ep['train_loss']:.4f}\n"
                    f"  Train Precision: {ep['train_precision']:.4f}\n"
                    f"  Train Recall: {ep['train_recall']:.4f}\n"
                    f"  Train F1: {ep['train_f1']:.4f}\n"
                    f"  Train Accuracy: {ep['train_accuracy']:.4f}\n"
                    f"  Val Loss: {ep['val_loss']:.4f}\n"
                    f"  Val Precision: {ep['val_precision']:.4f}\n"
                    f"  Val Recall: {ep['val_recall']:.4f}\n"
                    f"  Val F1: {ep['val_f1']:.4f}\n"
                    f"  Val Accuracy: {ep['val_accuracy']:.4f}\n"
                )

                # --- SAVE TO CSV ---
                logger.log_csv(config_idx, params, fold_id, ep)

        config_idx += 1

if __name__ == "__main__":
    hyperparameter_grid = {
        "lr": [1e-5, 5e-6, 3e-6, 2e-5],
        "batch_size": [16, 32],
        "weight_decay": [0.0, 0.01, 0.1],
        "warmup_ratio": [0.03, 0.06],
    }

    run_grid_search()
