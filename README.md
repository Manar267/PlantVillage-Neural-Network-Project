# PlantVillage Neural Network Project

Course: AIE231 – Neural Networks  
Institution: Alamein International University  
Track: Image Classification  
Dataset: PlantVillage (`55,425` verified images across `39` classes in project split; TFDS catalog notes `54,303` images across `38` classes)

## Project status
Initial milestone documentation in progress. Dataset sources and version notes, class imbalance rationale (`Macro-F1`), team roles, and baseline results are documented below; full model training and ablations follow.

## Dataset & Criteria Check
- **Sources:** TensorFlow Datasets (`https://www.tensorflow.org/datasets/catalog/plant_village`), Mendeley (`https://data.mendeley.com/datasets/tywbtsjrjv/1`), and original paper (`https://arxiv.org/abs/1511.08060`).
- **Version & Scope Note:** TFDS lists `54,303` images across `38` classes (excluding `Background_without_leaves`), while our project split contains `55,425` verified RGB images across `39` classes with zero missing values or duplicates.
- **Criteria Compliance:** Meets course project criteria—multi-class RGB leaf disease classification, sufficient image volume (`55,425` images across `39` classes), and clear evaluation setup using `Macro-F1` and held-out test evaluation.

## Planned models
1. Custom CNN (built from scratch in Keras)
2. VGG16 with frozen convolutional base (ImageNet weights)
3. Fine-tuned VGG16 (top layers unfrozen with a small learning rate; 3-seed evaluation)

## Problem Formulation

Smallholder farmers and agricultural extension workers need a reliable, automated way to identify plant species and leaf diseases from smartphone leaf images before infections spread across fields. Given a single RGB leaf image, the model predicts one of the plant species/disease classes in the PlantVillage dataset. Wrong predictions carry asymmetric costs: a **false negative** (classifying a diseased leaf as healthy) can allow a contagious crop disease to spread and reduce yield, whereas a **false positive** (classifying a healthy leaf as diseased) can cause unnecessary pesticide use and extra cost.

- **Input shape:** `(224, 224, 3)` RGB image (or `(128, 128, 3)` during fast baseline/ablation runs so each training run stays under 30 minutes on a free Colab/Kaggle GPU).
- **Output:** Probability distribution over the target classes (`Softmax` output layer).
- **Loss function:** `CategoricalCrossentropy` (for one-hot encoded labels) or `SparseCategoricalCrossentropy` (for integer class labels).
- **Primary metric:** **Macro-F1** (alongside per-class Recall and Confusion Matrix).
- **Why this metric:** PlantVillage has class imbalance across classes, so overall accuracy can be dominated by larger classes and hide poor performance on smaller or visually similar disease classes. Macro-F1 gives equal weight to every class so errors on minority classes are clearly reflected.

## Baseline & Model Results Summary

| Model | Split | Accuracy | Macro F1 | Notes |
| :--- | :--- | :---: | :---: | :--- |
| Simple Learned Baseline (LogReg, 32x32) | Val | `0.5870` | `0.4574` | Sampled subset (3k train / 1k val) |
| Simple Learned Baseline (LogReg, 32x32) | Test | `0.5970` | `0.4944` | Sampled subset (1k test) |
| Model A: Custom CNN (from scratch) | Val / Test | `TBD` | `TBD` | Built from scratch in Keras |
| Model B: Pretrained CNN (Frozen Base) | Val / Test | `TBD` | `TBD` | Frozen VGG16 base (ImageNet weights) |
| Model C: Fine-Tuned Pretrained CNN | Val / Test | `TBD` | `TBD` | Fine-tuned top layers (3-seed evaluation) |

## Team Roles (5-Member Team)

| Member | Role(s) | Main Responsibilities |
| :--- | :--- | :--- |
| Rodina Mahmoud Adel Ghanem | Tech Lead & Baseline/Analysis Engineer (Merged Roles 1 & 3) | Problem formulation, repo coordination/reviews, final 3-seed run & summary table, EDA, baselines, and Model A comparison |
| Manar Samir (`Manar267`) | Experimentation Engineer | Repository setup & team coordination |
| Haneen Wael (`haneen-wael`) | Data Engineer | Repository files & pipeline tasks |
| Member 4 | *(Confirm assigned role)* | Assigned work package & neural experiment |
| Member 5 | *(Confirm assigned role)* | Assigned work package & neural experiment |

## Repository layout
- `data/`: local dataset files; dataset contents are not committed to Git.
- `notebooks/`: exploratory analysis.
- `src/data/`: data loading and preprocessing.
- `src/models/`: model architectures.
- `src/experiments/`: controlled experiment runner and logging.
- `configs/`: shared configuration.
- `results/`: generated metrics and figures.
- `models/`: saved model artifacts.
- `app/`: demonstration application.

## Setup
Create and activate a Python virtual environment, then install:
```bash
pip install -r requirements.txt
