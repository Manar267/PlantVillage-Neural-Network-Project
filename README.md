# PlantVillage Neural Network Project

Course: AIE231 – Neural Networks  
Institution: Alamein International University  
Track: Image Classification  
Dataset: PlantVillage (54,303 images; 38 classes, subject to verification against the selected download)

## Project status
Initial repository scaffold. Dataset version, license, split strategy, preprocessing decisions, and model configurations must be documented after inspection.

## Planned models
1. Custom CNN
2. VGG16 with frozen convolutional base
3. Fine-tuned VGG16
## Problem Formulation

Smallholder farmers and agricultural extension workers need a reliable, automated way to identify plant species and leaf diseases from smartphone leaf images before infections spread across fields. Given a single RGB leaf image, the model predicts one of the 38 plant species/disease classes in the PlantVillage dataset. Wrong predictions carry asymmetric costs: a **false negative** (classifying a diseased leaf as healthy) can allow a contagious crop disease to spread and reduce yield, whereas a **false positive** (classifying a healthy leaf as diseased) can cause unnecessary pesticide use and extra cost.

- **Input shape:** `(224, 224, 3)` RGB image (or `(128, 128, 3)` during fast baseline/ablation runs so each training run stays under 30 minutes on a free Colab/Kaggle GPU).
- **Output:** Probability distribution over the 38 classes (`Softmax` output layer).
- **Loss function:** `CategoricalCrossentropy` (for one-hot encoded labels) or `SparseCategoricalCrossentropy` (for integer class labels).
- **Primary metric:** **Macro-F1** (alongside per-class Recall and Confusion Matrix).
- **Why this metric:** PlantVillage has class imbalance across its 38 classes, so overall accuracy can be dominated by larger classes and hide poor performance on smaller or visually similar disease classes. Macro-F1 gives equal weight to every class so errors on minority classes are clearly reflected.

## Team Roles (5-Member Team)

| Member | Role(s) | Main Responsibilities |
| :--- | :--- | :--- |
| Rodina Mahmoud Adel Ghanem | Tech Lead & Baseline/Analysis Engineer (Merged Roles 1 & 3) | Problem formulation, repo coordination/reviews, final 3-seed run & summary table, EDA, baselines, and Model A comparison |
| Manar Samir (`Manar267`) |  Experimentation 
Engineer  | Repository setup & team coordination |
| Haneen Wael (`haneen-wael`) |  Data Engineer  | Repository files & pipeline tasks |
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
```

The exact dataset download and training commands will be added after the team confirms the dataset source/version and data format.

## Reproducibility
All experiments should record the model, data split/version, random seed, optimizer, learning rate, batch size, epochs, and validation metrics. The test set must be held out until final evaluation.
