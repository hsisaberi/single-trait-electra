import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ContextualAugmentor:
    def __init__(self, model_name="google/gemma-3-27b-it"):
        print("Loading Gemma-27B-IT model... This may take some time.")

        self.device = "cuda:4"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": self.device}).to(self.device)
    
    def augment_text(self, text):
        prompt = (
            "Rewrite the following essay with different wording while preserving the meaning, "
            "style, and emotional tone. Do NOT shorten it too much.\n\n"
            f"Essay:\n{text}\n\nParaphrased Essay:"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=10000, temperature=0.0)

        generated = self.tokenizer.decode(output[0], skip_special_tokens=True)

        # Extract only the paraphrased part (everything after the prompt)
        if "Paraphrased Essay:" in generated:
            generated = generated.split("Paraphrased Essay:")[1].strip()

        return generated
