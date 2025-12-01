import os
import nltk
import random
import pandas as pd
from tqdm import tqdm
from utils import load_split_dataset
from augmentation import *

def synonym_replacor(df):
    df = df.copy()
    df.insert(0, "id", range(len(df)))  # id as the first column

    augmented_rows = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing Rows for Synonym_Replacement"):
        text = row["text"]
        new_text = synonym_replace(text, n_per_sentence=2)

        new_row = row.copy()
        new_row["text"] = new_text
        augmented_rows.append(new_row)
    
    df_aug = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)
    df_aug = df_aug.rename(columns={"text": "TEXT"})        # For name consistency in training (should be solved later)
    return df_aug

def contextual_augmentor(df):
    augmentor = ContextualAugmentor()

    augmented_rows = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Contextual Augmentation"):
        original_text = row["text"]
        new_text = augmentor.augment_text(original_text)

        new_row = row.copy()
        new_row["text"] = new_text
        augmented_rows.append(new_row)

    df_aug = pd.concat([df, pd.DataFrame(augmented_rows)], ignore_index=True)

    return df_aug

def main(aug_type):
    project_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(project_dir, "dataset")
    file_path = os.path.join(dataset_path, "essay_training_with_id.csv")
    df = load_split_dataset(file_path, "All", split=False)

    if aug_type == "synonym":
        output_file_path = os.path.join(dataset_path, "augmentation/synonym_replacement/augmented_syn.csv")
        df_aug = synonym_replacor(df)
    
    elif aug_type == "contextual":
        output_file_path = os.path.join(dataset_path, "augmentation/contextual_augmentation/augmented_contextual.csv")
        df_aug = contextual_augmentor(df)

    else:
        print("This option is not implemented")

    df_aug.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    main(aug_type="synonym")
