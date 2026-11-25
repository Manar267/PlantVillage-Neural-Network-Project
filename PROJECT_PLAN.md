# Project Plan

## Working title
PlantVillage Leaf Disease Classification

## Course
AIE231 – Neural Networks, Alamein International University

## Current decisions (provisional)
- Track: Image Classification
- Dataset: PlantVillage
- Task: 38-class classification of leaf images
- Framework: TensorFlow / Keras
- Planned models: Custom CNN, frozen VGG16, fine-tuned VGG16
- Main development environment: To be confirmed

## Decisions to verify
- Exact dataset source, version, download URL, and license
- Actual image count and per-class distribution in the downloaded copy
- Whether duplicate/near-duplicate images exist
- Train/validation/test split method and random seed
- Image dimensions, color mode, and preprocessing
- Primary metric and supporting metrics
- Team members, roles, and ownership

## Team
| Member | Role | Owned files / deliverables |
|---|---|---|
| TBD | Tech Lead | Repository, integration, final model selection |
| TBD | Data Engineer | Data loading and preprocessing |
| TBD | Baseline & Analysis Engineer | EDA and baselines |
| TBD | Model Architect | Model architectures |
| TBD | Experimentation Engineer | Experiment runner and tuning studies |
| TBD | Evaluation & Deployment Engineer | Evaluation and demo |

## Important project rules
- Do not commit dataset images or large model checkpoints to Git.
- Fit preprocessing transformations using training data only.
- Keep the test set untouched during model selection and tuning.
- Record experiment settings and results.
- Document dataset source and license.
