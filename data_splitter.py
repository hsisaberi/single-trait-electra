import os
import pandas as pd
from utils import load_split_dataset
from sklearn.model_selection import train_test_split

def splitter(data_path, save_path, train_ratio):
    df = pd.read_csv(data_path)
    x = df['TEXT']
    y = df[['cEXT', 'cNEU', 'cAGR', 'cCON', 'cOPN']]

    test_ratio = 1 - train_ratio
    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=test_ratio, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, random_state=42)

    train_data = pd.concat([x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    val_data = pd.concat([x_val.reset_index(drop=True), y_val.reset_index(drop=True)], axis=1)
    test_data = pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

    train_data.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    val_data.to_csv(os.path.join(save_path, 'validation.csv'), index=False)
    test_data.to_csv(os.path.join(save_path, 'test.csv'), index=False)

def split_augmented():
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_dir, "dataset/augmentation/contextual_augmentation/augmented_contextual.csv")
    save_path = os.path.join(project_dir, "dataset/split")

    df_train, df_val, df_test = load_split_dataset(data_path, trait_name="All", split=True)
    df_train.to_csv(os.path.join(save_path, 'train.csv'), index=False)
    df_val.to_csv(os.path.join(save_path, 'validation.csv'), index=False)
    df_test.to_csv(os.path.join(save_path, 'test.csv'), index=False)

if __name__ == "__main__":
    split_augmented()
