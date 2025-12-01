import os
import torch
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification, ElectraConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import *

class CrossValidationTrainer:
    def __init__(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Dataset
        self.csv_path = "./dataset/augmentation/contextual_augmentation/augmented_contextual.csv"
        self.trait_name = "Openness"
        self.k_folds = 10

        # Model
        self.model_name = "google/electra-base-discriminator"
        self.tokenizer = ElectraTokenizerFast.from_pretrained(self.model_name)

        # Hyperparameters
        self.epochs = 10
        self.lr = 5e-6
        self.batch_size = 16
        self.weight_decay = 0.01
        self.warmup_ratio = 0.06

        self.device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

        # Save Models
        self.save_dir = f"./model/ZFold_CrossValidation"

    def create_model(self, num_training_steps):
        config = ElectraConfig.from_pretrained(
            self.model_name,
            num_labels=2,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )

        model = ElectraForSequenceClassification.from_pretrained(self.model_name, config=config).to(self.device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        warmup_steps = int(self.warmup_ratio * num_training_steps)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

        return model, optimizer, scheduler

    def evaluate(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                
                preds = torch.argmax(outputs.logits, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        return {
            "accuracy": accuracy_score(all_labels, all_preds),
            "precision": precision_score(all_labels, all_preds, zero_division=0),
            "recall": recall_score(all_labels, all_preds, zero_division=0),
            "f1": f1_score(all_labels, all_preds, zero_division=0)
        }

    def compute_loss(self, model, loader):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                total_loss += outputs.loss.item()

        return total_loss / len(loader)

    def train_fold(self, fold, train_df, val_df):
        print(f"\n===== Fold {fold+1}/{self.k_folds} =====\n")

        # TensorBoard
        log_dir = f"./logs/{self.trait_name}_Fold_{fold+1}_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir)

        # Dataset
        train_ds = EssayDataset(train_df, self.tokenizer)
        val_ds = EssayDataset(val_df, self.tokenizer)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        # Model + optimizer + scheduler
        total_steps = len(train_loader) * self.epochs
        model, optimizer, scheduler = self.create_model(total_steps)

        # Training loop per epoch
        for epoch in range(1, self.epochs + 1):
            model.train()
            total_loss = 0
            train_preds, train_labels = [], []

            loop = tqdm(train_loader, desc=f"Training Fold {fold+1} Epoch {epoch}")

            for batch in loop:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                train_preds.extend(preds.cpu().tolist())
                train_labels.extend(batch["labels"].cpu().tolist())

                loop.set_postfix(loss=loss.item())

            # Train metrics
            train_loss = total_loss / len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)
            train_prec = precision_score(train_labels, train_preds, zero_division=0)
            train_rec = recall_score(train_labels, train_preds, zero_division=0)
            train_f1 = f1_score(train_labels, train_preds, zero_division=0)

            # Validation
            val_loss = self.compute_loss(model, val_loader)
            val_metrics = self.evaluate(model, val_loader)

            # Log to TensorBoard
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)

            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Val", val_metrics["accuracy"], epoch)

            writer.add_scalar("Precision/Train", train_prec, epoch)
            writer.add_scalar("Precision/Val", val_metrics["precision"], epoch)

            writer.add_scalar("Recall/Train", train_rec, epoch)
            writer.add_scalar("Recall/Val", val_metrics["recall"], epoch)

            writer.add_scalar("F1/Train", train_f1, epoch)
            writer.add_scalar("F1/Val", val_metrics["f1"], epoch)

            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)

            # Print results
            print(f"\n--- Fold {fold+1} Epoch {epoch} ---")
            print(f"Train Loss:      {train_loss:.4f}")
            print(f"Train Acc:       {train_acc:.4f}")
            print(f"Train F1:        {train_f1:.4f}")

            print(f"Val Loss:        {val_loss:.4f}")
            print(f"Val Acc:         {val_metrics['accuracy']:.4f}")
            print(f"Val F1:          {val_metrics['f1']:.4f}")

        writer.close()
        return val_metrics, model

    def run(self):
        df = load_split_dataset(self.csv_path, trait_name=self.trait_name, split=False)

        kf = KFolder(n_splits=self.k_folds, shuffle=True, random_state=42)

        fold_results = []

        # Loop through folds
        for train_df, val_df, fold_idx in kf.kfold_split(df):
            metrics, model = self.train_fold(fold_idx, train_df, val_df)

            fold_results.append(metrics)

            save_path = f"{self.save_dir}/{self.trait_name}_Fold_{fold_idx+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved model for fold {fold_idx+1} → {save_path}")

        # Print final statistics
        print("\n========== K-Fold Cross-Validation Results ==========")
        for metric in ["accuracy", "precision", "recall", "f1"]:
            avg = np.mean([f[metric] for f in fold_results])
            std = np.std([f[metric] for f in fold_results])
            print(f"{metric.capitalize()}: {avg:.4f} ± {std:.4f}")
        print("=====================================================\n")


if __name__ == "__main__":
    trainer = CrossValidationTrainer()
    trainer.run()
