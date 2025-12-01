import os
import torch
from utils import load_split_dataset
from transformers import ElectraTokenizerFast, ElectraForSequenceClassification

class Inference:
    def __init__(self):
        # Config
        self.device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu") 
        self.model_name = "google/electra-base-discriminator"
        self.tokenizer = ElectraTokenizerFast.from_pretrained(self.model_name)

        # Paths
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_paths = {
            "Openness": os.path.join(self.current_dir, "model/Openness/best.pt"),
            "Agreeableness": os.path.join(self.current_dir, "model/Agreeableness/best.pt"),
            "Conscientiousness": os.path.join(self.current_dir, "model/Conscientiousness/best.pt"),
            "Extroversion": os.path.join(self.current_dir, "model/Extroversion/best.pt"),
            "Neuroticism": os.path.join(self.current_dir, "model/Neuroticism/best.pt"),
        }
        
        # Load all 5 models
        self.models = {}
        for trait, path in self.model_paths.items():
            model = ElectraForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.models[trait] = model

    def predict_single(self, text):
        """ Return predictions for all 5 personality traits."""
        # Tokenize
        encoding = self.tokenizer(text, truncation=True, max_length=128, padding="max_length", return_tensors="pt").to(self.device)

        results = {}
        for trait, model in self.models.items():
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)[0]

                label = torch.argmax(probs).item()
                probability = probs[label].item()

                results[trait] = {
                    "label": label,           # 0 or 1
                    "confidence": probability # probability of predicted class
                }

        return results

if __name__ == "__main__":
    inf = Inference()
    test_data_path = os.path.join(inf.current_dir, "dataset/split/test.csv")
    data_df = load_split_dataset(test_data_path, trait_name="All")
    text = data_df['TEXT'].iloc[0]
    
    # or you can write a text
    # text = "Hello World!"

    result = inf.predict_single(text)
    print(result)
