import os
import torch
from torch.utils.data import DataLoader
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import load_split_dataset, EssayDataset, figures

class Evaluation:
    def __init__(self):
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-discriminator")

    def load_model(self, trait_name):
        trait_paths = {
            "Openness": "model/Openness/best.pt",
            "Neuroticism": "model/Neuroticism/best.pt",
            "Extroversion": "model/Extroversion/best.pt",
            "Conscientiousness": "model/Conscientiousness/best.pt",
            "Agreeableness": "model/Agreeableness/best.pt",
        }

        if trait_name not in trait_paths:
            raise ValueError(f"Unknown trait: {trait_name}")

        model_path = os.path.join(self.project_dir, trait_paths[trait_name])

        model = ElectraForSequenceClassification.from_pretrained("google/electra-base-discriminator", num_labels=2)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()

        return model

    def evaluate_model(self, model, loader):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
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

                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)[:, 1]
                preds = torch.argmax(logits, dim=-1)

                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(batch["labels"].cpu().tolist())

        avg_loss = total_loss / len(loader)
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        conf_matrix = confusion_matrix(all_labels, all_preds)

        return avg_loss, accuracy, precision, recall, f1, conf_matrix, all_probs, all_labels


    def run_eval(self, model, trait_name, data_path, dataset_name, batch_size=16):
        df = load_split_dataset(data_path, trait_name, split=False)
        ds = EssayDataset(df, self.tokenizer)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        loss, acc, prec, rec, f1, conf_matrix, probs, labels = self.evaluate_model(model, loader)

        print(f"\n===== Evaluation: {os.path.basename(data_path)} =====")
        print(f"Loss:       {loss:.4f}")
        print(f"Accuracy:   {acc:.4f}")
        print(f"Precision:  {prec:.4f}")
        print(f"Recall:     {rec:.4f}")
        print(f"F1 Score:   {f1:.4f}")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("=============================================")

        return (dataset_name, probs, labels)

if __name__ == "__main__":
    trait_name = "Openness"

    train_data = "./dataset/split/train.csv"
    validation_data = "./dataset/split/validation.csv"
    test_data = "./dataset/split/test.csv"

    evaluator = Evaluation()

    # Load model only once
    model = evaluator.load_model(trait_name)

    results = []
    results.append(evaluator.run_eval(model, trait_name, train_data, "Train"))
    results.append(evaluator.run_eval(model, trait_name, validation_data, "Validation"))
    results.append(evaluator.run_eval(model, trait_name, test_data, "Test"))

    # Plot curves
    figures.plot_roc(results, trait_name, os.path.join(evaluator.project_dir, "result"))
    figures.plot_pr(results, trait_name, os.path.join(evaluator.project_dir, "result"))
