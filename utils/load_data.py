import pandas as pd
from sklearn.model_selection import train_test_split

def load_split_dataset(file_path, trait_name, split=True, val_size=0.2, random_state=42):
    df = pd.read_csv(file_path, encoding='latin1')

    if "TEXT" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"TEXT": "text"})

    if "text" not in df.columns:
        raise ValueError("No text column found. Expected 'TEXT' or 'text'.")
    
    # Map traits to columns
    trait_map = {
        "Openness": "cOPN",
        "Conscientiousness": "cCON",
        "Extroversion": "cEXT",
        "Agreeableness": "cAGR",
        "Neuroticism": "cNEU",
    }

    # Single-trait case
    if trait_name != "All":
        if trait_name not in trait_map:
            raise ValueError(f"Trait '{trait_name}' not available.")
        label_col = trait_map[trait_name]
        if label_col not in df.columns:
            raise ValueError(f"Column '{label_col}' not found.")
        
        df = df[["id", "text", label_col]].rename(columns={label_col: "label"})
        df["label"] = df["label"].astype(int)

    else:
        # Multi-trait: keep all trait columns + text + id
        required_cols = ["id", "text"] + list(trait_map.values())
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        df = df[required_cols]

    if not split:
        return df

    # Original rows: first occurrence of each id
    df_original = df.drop_duplicates(subset="id", keep="first")

    # Train / temp split (val+test)
    df_train_orig, df_temp = train_test_split(df_original, test_size=val_size, random_state=random_state, shuffle=True)

    # Split temp into validation and test
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=random_state, shuffle=True)

    # Final training set
    df_train_final = df[df["id"].isin(df_train_orig["id"])].reset_index(drop=True)

    return df_train_final, df_val.reset_index(drop=True), df_test.reset_index(drop=True)

if __name__ == "__main__":
    # Example usage
    file_path = "./dataset/augmentation/synonym_replacement/augmented_syn.csv"
    
    # df = load_split_dataset(file_path, "Openness", split=False)
    # print(df.head())

    df_train, df_val, df_test = load_split_dataset(file_path, "All", split=True)
    print(df_train.head())
