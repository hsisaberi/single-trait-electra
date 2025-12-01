import os
import pandas as pd
from utils import barplot

def ditribution_plot(data_path, category):
    train = pd.read_csv(data_path)
    agreeableness = train["cAGR"].to_numpy()
    extraversion = train["cEXT"].to_numpy()
    conscientiousness = train["cCON"].to_numpy()
    neuroticism = train["cNEU"].to_numpy()
    openness = train["cOPN"].to_numpy()    

    barplot(agreeableness, "Agreeableness", save_dir, category)
    barplot(extraversion, "Extroversion", save_dir, category)
    barplot(conscientiousness, "Conscientiousness", save_dir, category)
    barplot(neuroticism, "Neuroticism", save_dir, category)
    barplot(openness, "Openness", save_dir, category)
    
if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(project_dir, "dataset/split")
    
    # Train
    train_path = os.path.join(project_dir, "dataset/split/train.csv")
    ditribution_plot(train_path, "Train")
    
    # Test
    test_path = os.path.join(project_dir, "dataset/split/test.csv")
    ditribution_plot(test_path, "Test")

    # Validation
    validation_path = os.path.join(project_dir, "dataset/split/validation.csv")
    ditribution_plot(validation_path, "Validation")
