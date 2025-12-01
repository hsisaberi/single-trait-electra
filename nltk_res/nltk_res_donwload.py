import os
import nltk

local_nltk_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(local_nltk_dir, exist_ok=True)

nltk.data.path.clear()
nltk.data.path.append(local_nltk_dir)

nltk.download('punkt', download_dir=local_nltk_dir)
nltk.download("punkt_tab", download_dir=local_nltk_dir)
nltk.download('stopwords', download_dir=local_nltk_dir)
nltk.download('wordnet', download_dir=local_nltk_dir)
nltk.download('averaged_perceptron_tagger', download_dir=local_nltk_dir)
nltk.download('averaged_perceptron_tagger_eng', download_dir=local_nltk_dir)
nltk.download('omw-1.4', download_dir=local_nltk_dir)
