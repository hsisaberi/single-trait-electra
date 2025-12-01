# Transformer-Based Personality Trait Recognition

### Using ELECTRA + Data Augmentation + K-Fold Cross Validation

This repository contains the full implementation of the experiments described in the manuscript:

> ***“Transformer-Based Personality Trait Recognition Enhanced by Contextual Augmentation”***
> Saberi & Ravanmehr, 2025 

The project provides a complete pipeline for **data preprocessing**, **augmentation**, **model training**, **k-fold cross-validation**, **grid search**, **evaluation**, and **inference** for the **Big Five Personality Traits** using **ELECTRA-based classifiers**.

Each personality trait, **Openness**, **Conscientiousness**, **Extraversion**, **Agreeableness**, **Neuroticism**, is modeled independently following the single-trait design described in the manuscript.

---

## Repository Overview

```
├── augment.py
├── data_analyzer.py
├── data_splitter.py
├── eval.py
├── infer.py
├── train.py
├── kfold_train.py
├── utils/
├── nltk_res/
│   └── nltk_res_download.py
├── grid_search/
│   ├── grid_search.py
│   └── grid_search_results_results/   # automatically generated during grid search
├── dataset/
│   ├── essay_training.csv       # original dataset
│   ├── folds/                   # 5-fold cross validation splits
│   ├── split/                   # standard train/val/test split
│   └── augmentation/            # synonym + contextual augmentation results
```

---

## Installation

### 1. Create the Conda Environment (Python 3.11)

```bash
conda env create --name personality python=3.11
conda activate personality
```

### 2. Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install NLTK Resources (WordNet, tokenizers, etc.)

```bash
python nltk_res/nltk_res_download.py
```

---


## Dataset Description

This repository uses the **Pennebaker & King Essays dataset**, containing 2,467 essays labeled with the **Big Five** personality traits. Two augmented version is created through **WordNet synonym replacement** and **contextual augmentation using gemma-27b-it**, consistent with the manuscript methodology.
All processed data (splits, folds, augmented essays) are located in `./dataset/`.

---

## 1. Data Splitting

Before any training, you must create the dataset split:

```bash
python data_splitter.py
```

## 2. Training a Model (Single-Trait)

Once the split is created, inside the training script, set your target trait, for example:

```python
self.trait_name = "Openness"
```

```bash
python train.py
```

This trains **one ELECTRA classifier** for the selected trait.
Repeat for all five traits by changing `self.trait_name` accordingly.

---

## 3. Training With K-Fold Cross-Validation

To reproduce the manuscript’s k-fold training experiments:

```bash
python k-fold_train.py
```

This uses the pre-generated 5 folds located in:

```
./dataset/folds/
```

---

## 4. Grid Search (Hyperparameter Optimization)

Grid search over ELECTRA hyperparameters using k-fold CV:

```bash
python grid_search/grid_search.py
```

Grid search results are saved automatically to:

```
./grid_search/
```

---

## 5. Evaluation

To evaluate any trained model on the test set:

```bash
python eval.py
```

This computes accuracy, precision, recall, F1, ROC-AUC, and PR-AUC (as described in the manuscript).

---

## 6. Inference

For running inference on new text samples using the trained model:

```bash
python infer.py
```

You must load any trait-specific checkpoint to generate a predicted high/low label.

---

## 7. Data Augmentation

To apply the augmentation pipeline:

```bash
python augment.py
```

This produces augmented essays using:

* **Synonym Replacement (WordNet)**
* **Contextual Augmentation (Gemma-based)** 

Augmented outputs populate:

```
./dataset/augmentation/
```

---

## 8. NLTK Resource Downloader

If NLTK resources are missing:

```bash
python nltk_res/nltk_res_download.py
```

This downloads WordNet, punkt, averaged_perceptron_tagger, etc.

---

## Manuscript Summary

This repository implements the full experimental pipeline from the paper, which developed **five independent ELECTRA-based binary classifiers**, one per personality trait, achieving:

* **Average Accuracy**: ~72.4%
* **Test AUC Scores**: >0.75 for all traits
* **Dataset Expanded**: 2,467 → 4,934 samples (synonym + contextual augmentation)

Using a trait-isolated approach reduces cross-trait interference and improves generalization, as detailed in the manuscript.
All training/evaluation curves, PR/ROC results, and metrics are reproducible through the scripts in this repo.


---

## Citation

If you use this repository, please cite the associated manuscript:

```
Saberi & Ravanmehr. 
"Transformer-Based Personality Trait Recognition Enhanced by Contextual Augmentation" (2025).
```

---

## 🤝 Contributions

Contributions, suggestions, or improvements are welcome.
Feel free to open issues or pull requests.

---