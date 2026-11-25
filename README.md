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
