"""Dataset loading and preprocessing.

Implement after the team confirms the exact PlantVillage source and folder format.
Keep split creation and preprocessing fitted on training data only.
"""

def load_dataset(*args, **kwargs):
    raise NotImplementedError("Implement after confirming the dataset format.")

def create_splits(*args, **kwargs):
    raise NotImplementedError("Implement a reproducible, stratified split.")

def build_preprocessing_pipeline(*args, **kwargs):
    raise NotImplementedError("Implement resizing, scaling, and augmentation.")
