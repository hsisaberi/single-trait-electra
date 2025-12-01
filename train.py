import os
import torch
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification, ElectraConfig
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from utils import *

class Train:
    def __init__(self):
        project_dir = os.path.dirname(os.path.abspath(__file__))

        # Pahts
        self.trait_name = "Openness"

        self.train_path = os.path.join(project_dir, "dataset/split/train.csv")
        self.validation_path = os.path.join(project_dir, "dataset/split/validation.csv")
        self.test_path = os.path.join(project_dir, "dataset/split/test.csv")

        self.model_path = os.path.join(project_dir, f"model/{self.trait_name}")
        os.makedirs(self.model_path, exist_ok=True)

        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

        # Load tokenizer + model config
        self.model_name = "google/electra-base-discriminator"
        self.tokenizer = ElectraTokenizerFast.from_pretrained(self.model_name)
        config = ElectraConfig.from_pretrained(
            self.model_name,
            num_labels=2,
            hidden_dropout_prob=0.1,                # default
            attention_probs_dropout_prob=0.1,       # default
        )
        self.model = ElectraForSequenceClassification.from_pretrained(self.model_name, config=config).to(self.device)
        
        # Config
        self.epoch = 10
        self.lr = 8e-06
        self.batch_size = 16
        self.weight_decay = 0.0
        warmup_ratio = 0.06

        train_df = load_split_dataset(self.train_path, trait_name=self.trait_name, split=False)
        total_steps = (len(train_df) // self.batch_size) * self.epoch
        warmup_steps = int(warmup_ratio * total_steps)       # 10% of total steps
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # Tensorboard log directory
        log_dir = os.path.join(os.path.join(project_dir, "logs"), datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        print(f"Tensorboard logging at: {log_dir}")

    def data_loader(self):
        train_df = load_split_dataset(self.train_path, trait_name=self.trait_name, split=False)
        train_ds = EssayDataset(train_df, self.tokenizer)

        validation_df = load_split_dataset(self.validation_path, trait_name=self.trait_name, split=False)
        validation_ds = EssayDataset(validation_df, self.tokenizer)

        test_df = load_split_dataset(self.test_path, trait_name=self.trait_name, split=False)
        test_ds = EssayDataset(test_df, self.tokenizer)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_ds, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)

        return train_loader, validation_loader, test_loader

    def evaluate(self, model, loader):
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                
                preds = torch.argmax(outputs.logits, dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())
        
        # Compute metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def compute_loss(self, model, loader):
        model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
                total_loss += outputs.loss.item()
        
        return total_loss/len(loader) 

    def save_model(self, epoch):
        save_path = os.path.join(self.model_path, f"model_epoch_{epoch}.pt")
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved: {save_path}")

    def train(self):
        train_loader, validation_loader, test_loader = self.data_loader()

        history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "test_accuracy": []
        }

        for epoch in range(1, self.epoch + 1):
            self.model.train()
            total_loss = 0
            train_preds, train_labels = [], []

            loop = tqdm(train_loader, desc=f"[Epoch {epoch}] Training")
            for batch in loop:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self.optimizer.zero_grad()
                
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )

                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)    # gradient clipping
                self.optimizer.step()
                self.scheduler.step()

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

                preds = torch.argmax(outputs.logits, dim=-1)
                train_preds.extend(preds.cpu().tolist())
                train_labels.extend(batch["labels"].cpu().tolist())
            
            # Train Metrics
            train_loss = total_loss / len(train_loader)
            train_accuracy = accuracy_score(train_labels, train_preds)
            train_precision = precision_score(train_labels, train_preds, zero_division=0)
            train_recall = recall_score(train_labels, train_preds, zero_division=0)
            train_f1 = f1_score(train_labels, train_preds, zero_division=0)
            
            # Log train metrics
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_accuracy, epoch)
            self.writer.add_scalar("Precision/Train", train_precision, epoch)
            self.writer.add_scalar("Recall/Train", train_recall, epoch)
            self.writer.add_scalar("F1/Train", train_f1, epoch)
            
            # validation
            val_loss = self.compute_loss(self.model, validation_loader)
            val_metrics = self.evaluate(self.model, validation_loader)
           
            # Log val metrics
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("Accuracy/Validation", val_metrics["accuracy"], epoch)
            self.writer.add_scalar("Precision/Validation", val_metrics["precision"], epoch)
            self.writer.add_scalar("Recall/Validation", val_metrics["recall"], epoch)
            self.writer.add_scalar("F1/Validation", val_metrics["f1"], epoch)

            # Test
            test_loss = self.compute_loss(self.model, test_loader)
            test_metrics = self.evaluate(self.model, test_loader)

            # Log test metrics
            self.writer.add_scalar("Loss/Test", test_loss, epoch)
            self.writer.add_scalar("Accuracy/Test", test_metrics["accuracy"], epoch)
            self.writer.add_scalar("Precision/Test", test_metrics["precision"], epoch)
            self.writer.add_scalar("Recall/Test", test_metrics["recall"], epoch)
            self.writer.add_scalar("F1/Test", test_metrics["f1"], epoch)

            # Log LR
            self.writer.add_scalar("LR", self.scheduler.get_last_lr()[0], epoch)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)

            history["train_accuracy"].append(train_accuracy)
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["test_accuracy"].append(test_metrics["accuracy"])

            # Print results
            print(f"\n===== Epoch {epoch} Results =====")

            print("\n--- Train ---")
            print(f"Train Loss:      {train_loss:.4f}")
            print(f"Train Accuracy:  {train_accuracy:.4f}")
            print(f"Train Precision: {train_precision:.4f}")
            print(f"Train Recall:    {train_recall:.4f}")
            print(f"Train F1 Score:  {train_f1:.4f}")

            print("\n--- Validation ---")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Accuracy:        {val_metrics['accuracy']:.4f}")
            print(f"Precision:       {val_metrics['precision']:.4f}")
            print(f"Recall:          {val_metrics['recall']:.4f}")
            print(f"F1 Score:        {val_metrics['f1']:.4f}")

            print("\n--- Test ---")
            print(f"Test Loss:       {test_loss:.4f}")
            print(f"Accuracy:        {test_metrics['accuracy']:.4f}")
            print(f"Precision:       {test_metrics['precision']:.4f}")
            print(f"Recall:          {test_metrics['recall']:.4f}")
            print(f"F1 Score:        {test_metrics['f1']:.4f}")

            print("=================================\n")

            self.save_model(epoch)

        self.writer.close()

        figures.plot_loss(
            history["train_loss"], 
            history["val_loss"], 
            history["test_loss"], 
            trait_name=self.trait_name, 
            save_dir=f"./model/{self.trait_name}"
        )

        figures.plot_accuracy(
            history["train_accuracy"],
            history["val_accuracy"],
            history["test_accuracy"],
            trait_name=self.trait_name,
            save_dir=f"./model/{self.trait_name}"
        )

if __name__ == "__main__":
    trainer = Train()
    trainer.train()
