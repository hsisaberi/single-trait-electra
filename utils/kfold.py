import os
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state

class KFolder():
    def __init__(self, n_splits, shuffle=False, random_state=None, save_dir="dataset/folds"):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def kfold_split(self, df):
        """
        Perform k-fold splitting with augmentation-safe logic.
        Validation = only original samples (one per id).
        Training = original + augmented samples of training ids
        Yields: train_df, val_df, fold_index
        """
        df_original = df.drop_duplicates(subset="id", keep="first")
        df_aug_only = df[df.duplicated(subset="id", keep="first")]

        n_samples = len(df_original)
        indices = np.arange(n_samples)

        # Shuffle if needed
        if self.shuffle:
            rng = check_random_state(self.random_state)
            rng.shuffle(indices)

        # fold sizes same logic as sklearn
        fold_sizes = np.full(self.n_splits, n_samples // self.n_splits, dtype=int)
        fold_sizes[: n_samples % self.n_splits] += 1

        # Generate folds
        current = 0
        for fold_idx, fold_size in enumerate(fold_sizes):
            start, stop = current, current + fold_size
            val_indices = indices[start:stop]
            train_indices = np.concatenate([indices[:start], indices[stop:]])
            current = stop

            # original splits
            val_orig = df_original.iloc[val_indices]
            train_orig = df_original.iloc[train_indices]

            # IDs
            val_ids = set(val_orig["id"].tolist())
            train_ids = set(train_orig["id"].tolist())

            # augmented rows for training only
            train_aug = df_aug_only[df_aug_only["id"].isin(train_ids)]

            # final train = original + augmented
            train_final = pd.concat([train_orig, train_aug], ignore_index=True).reset_index(drop=True)

            # final val = only originals
            val_final = val_orig.reset_index(drop=True)

            # Save to disk
            fold_path = os.path.join(self.save_dir, f"fold_{fold_idx+1}")
            os.makedirs(fold_path, exist_ok=True)

            train_final.to_csv(os.path.join(fold_path, "train.csv"), index=False)
            val_final.to_csv(os.path.join(fold_path, "val.csv"), index=False)

            print(f"Saved Fold {fold_idx+1}:")
            print(f"  Train → {fold_path}/train.csv (orig + aug)")
            print(f"  Val   → {fold_path}/val.csv  (original only)")

            yield train_final, val_final, fold_idx

if __name__ == "__main__":
    # Example usage
    data_path = "./dataset/augmentation/synonym_replacement/augmented_syn.csv"
    df = pd.read_csv(data_path)

    kf = KFolder(
        n_splits=5,
        shuffle=False,
        random_state=42,
        save_dir="dataset/folds"
    )

    for train_df, val_df, fold_idx in kf.kfold_split(df):
        print(f"Fold {fold_idx} → Train: {len(train_df)}, Val: {len(val_df)}")
