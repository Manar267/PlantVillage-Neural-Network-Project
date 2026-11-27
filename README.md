# PlantVillage Leaf Disease Classification (AIE231 Project)

- **Course:** AIE231 – Neural Networks
- **Institution:** Alamein International University
- **Track:** Image Classification

## 1. Problem Statement
This project addresses multi-class plant disease and species classification from leaf photographs using neural networks in TensorFlow/Keras.
- **Target Users:** Farmers, agronomists, and agricultural extension workers needing visual screening of crop leaf health.
- **Prediction & Error Cost:** Given a leaf image, the model predicts one of **39 classes** across **55,425** total images (`38,797` train, `8,314` validation, and `8,314` test). False negatives on crop diseases can delay intervention and cause yield loss, whereas false positives can cause unnecessary treatment costs. Because classes are imbalanced, **Macro F1-score** (alongside per-class precision/recall and accuracy) is reported so minority classes are not masked by majority classes.

## 2. Dataset
- **Dataset:** PlantVillage leaf disease dataset (`55,425` verified images across `39` classes; zero missing values, zero duplicates, and zero split overlap).
- **Source Link:** `[Add exact dataset URL here]`
- **Local Data Setup:** Raw images are stored outside Git and referenced via relative `image_path` entries. Configure the dataset root path (`IMAGES_ROOT`) as documented in `data/README.md`.

## 3. Planned Models & Baselines
- **Baselines:**
  1. Trivial Baseline (majority-class prediction)
  2. Simple Learned Baseline (logistic/linear classifier on downsampled image features)
- **Neural Networks (Keras):**
  - **Model A:** Custom CNN built from scratch
  - **Model B:** Pretrained VGG16 with frozen base and a new classification head (ImageNet weights)
  - **Model C:** Fine-tuned VGG16 (top layers unfrozen with a small learning rate; evaluated across at least 3 seeds)

## 4. Results Summary
Models are selected and tuned using validation metrics; final test metrics are evaluated once on the held-out test set.

| Model | Split | Accuracy | Macro F1 | Notes |
| :--- | :--- | :---: | :---: | :--- |
| Trivial Baseline (Majority Class) | Val | `0.0994` | `0.0046` | Always predicts `Orange___Haunglongbing_(Citrus_greening)` |
| Trivial Baseline (Majority Class) | Test | `0.0994` | `0.0046` | Majority-class reference benchmark |
| Simple Learned Baseline | Val / Test | `TBD` | `TBD` | Update once run completes |
| Model A: Custom CNN (from scratch) | Val / Test | `TBD` | `TBD` | Built from scratch in Keras |
| Model B: Pretrained CNN (Frozen Base) | Val / Test | `TBD` | `TBD` | Frozen VGG16 base (ImageNet weights) |
| Model C: Fine-Tuned Pretrained CNN | Val / Test | `TBD` | `TBD` | Fine-tuned top layers (report 3-seed mean ± std) |

## 5. Setup and Reproducibility
- **Random Seeds:** Fixed seeds are used across experiments (`SEED = 42` for development/ablations; at least 3 seeds for final evaluation).
- **Dependencies:** Package versions are pinned in `requirements.txt`.

### Install
```bash
pip install -r requirements.txt
