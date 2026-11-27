
import os
import json
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical


IMG_SIZE = 224
BATCH_SIZE = 32


# 1. Load Dataset

def load_dataset(dataset_path):

    classes = sorted([
        folder for folder in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, folder))
    ])

    image_paths = []
    labels = []

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)

        for filename in sorted(os.listdir(class_path)):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(class_path, filename))
                labels.append(class_name)

    dataframe = pd.DataFrame({
        'image_path': image_paths,
        'label': labels
    })

    return classes, dataframe

# 2. Check Corrupted Images

def check_corrupted_images(dataset_path, classes):

    corrupted_images = []
    total_checked = 0

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)

        for filename in sorted(os.listdir(class_path)):
            file_path = os.path.join(class_path, filename)

            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                total_checked += 1

                try:
                    with Image.open(file_path) as img:
                        img.verify()

                except Exception:
                    corrupted_images.append(file_path)

    return total_checked, corrupted_images


# 3. Find Duplicate Images

def find_duplicate_images(dataset_path, classes):

    image_hashes = defaultdict(list)

    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)

        for filename in sorted(os.listdir(class_path)):
            file_path = os.path.join(class_path, filename)

            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):

                with open(file_path, 'rb') as file:
                    image_hash = hashlib.md5(file.read()).hexdigest()

                image_hashes[image_hash].append(file_path)

    duplicate_groups = [
        paths for paths in image_hashes.values()
        if len(paths) > 1
    ]

    return image_hashes, duplicate_groups



# 4. Remove Duplicate Copies

def remove_duplicate_copies(dataframe, duplicate_groups):

    duplicate_paths_to_remove = {
        path
        for group in duplicate_groups
        for path in group[1:]
    }

    df_clean = dataframe[
        ~dataframe['image_path'].isin(duplicate_paths_to_remove)
    ].copy()

    return df_clean, duplicate_paths_to_remove


# 5. Split Dataset

def split_data(df_clean):

    train_df, temp_df = train_test_split(
        df_clean,
        test_size=0.30,
        random_state=42,
        stratify=df_clean['label']
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df['label']
    )

    return train_df, val_df, test_df


# 6. Encode Labels

def encode_labels(train_df, val_df, test_df):

    label_encoder = LabelEncoder()

    train_labels = label_encoder.fit_transform(
        train_df['label']
    )

    val_labels = label_encoder.transform(
        val_df['label']
    )

    test_labels = label_encoder.transform(
        test_df['label']
    )

    num_classes = len(label_encoder.classes_)

    y_train = to_categorical(
        train_labels,
        num_classes=num_classes
    )

    y_val = to_categorical(
        val_labels,
        num_classes=num_classes
    )

    y_test = to_categorical(
        test_labels,
        num_classes=num_classes
    )

    return (
        label_encoder,
        train_labels,
        val_labels,
        test_labels,
        y_train,
        y_val,
        y_test
    )


# 7. Image Preprocessing

def preprocess_image(image_path):

    image = tf.io.read_file(image_path)

    image = tf.image.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image = tf.image.convert_image_dtype(
        image,
        tf.float32
    )

    image = tf.image.resize(
        image,
        [IMG_SIZE, IMG_SIZE]
    )

    image = tf.ensure_shape(
        image,
        [IMG_SIZE, IMG_SIZE, 3]
    )

    return image


# 8. Create TensorFlow Dataset

def create_dataset(
    dataframe,
    labels,
    batch_size=BATCH_SIZE,
    shuffle=False
):

    dataset = tf.data.Dataset.from_tensor_slices(
        (dataframe['image_path'].values, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(dataframe),
            seed=42,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        lambda path, label: (
            preprocess_image(path),
            label
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# 9. Create Train, Validation and Test Datasets

def create_train_validation_test_datasets(
    train_df,
    val_df,
    test_df,
    y_train,
    y_val,
    y_test,
    batch_size=BATCH_SIZE
):

    train_dataset = create_dataset(
        train_df,
        y_train,
        batch_size,
        shuffle=True
    )

    val_dataset = create_dataset(
        val_df,
        y_val,
        batch_size
    )

    test_dataset = create_dataset(
        test_df,
        y_test,
        batch_size
    )

    return train_dataset, val_dataset, test_dataset


# 10. Calculate Class Weights

def calculate_class_weights(train_labels, num_classes):

    class_indices = np.arange(num_classes)

    weights = compute_class_weight(
        class_weight='balanced',
        classes=class_indices,
        y=train_labels
    )

    class_weights = {
        i: float(weight)
        for i, weight in enumerate(weights)
    }

    return class_weights


# 11. Add Sample Weights

def add_sample_weights(
    images,
    labels,
    class_weights_tensor
):

    class_ids = tf.argmax(
        labels,
        axis=1,
        output_type=tf.int32
    )

    sample_weights = tf.gather(
        class_weights_tensor,
        class_ids
    )

    return images, labels, sample_weights


# 12. Create Weighted Train Dataset

def create_weighted_train_dataset(
    train_df,
    y_train,
    class_weights,
    batch_size=BATCH_SIZE
):

    class_weights_tensor = tf.constant(
        [
            class_weights[i]
            for i in range(len(class_weights))
        ],
        dtype=tf.float32
    )

    train_dataset_weighted = tf.data.Dataset.from_tensor_slices(
        (train_df['image_path'].values, y_train)
    )

    train_dataset_weighted = train_dataset_weighted.shuffle(
        buffer_size=len(train_df),
        seed=42,
        reshuffle_each_iteration=True
    )

    train_dataset_weighted = train_dataset_weighted.map(
        lambda path, label: (
            preprocess_image(path),
            label
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset_weighted = train_dataset_weighted.batch(
        batch_size
    )

    train_dataset_weighted = train_dataset_weighted.map(
        lambda images, labels: add_sample_weights(
            images,
            labels,
            class_weights_tensor
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset_weighted = train_dataset_weighted.prefetch(
        tf.data.AUTOTUNE
    )

    return train_dataset_weighted


# 13. Create Data Augmentation Layers

def create_augmentation_layers():

    rotation_layer = tf.keras.layers.RandomRotation(
        factor=0.25,
        fill_mode='reflect'
    )

    zoom_layer = tf.keras.layers.RandomZoom(
        height_factor=(-0.15, 0.15),
        width_factor=(-0.15, 0.15),
        fill_mode='reflect'
    )

    return rotation_layer, zoom_layer


# 14. Augment Images

def augment_image(
    image,
    rotation_layer,
    zoom_layer
):

    image = rotation_layer(
        image,
        training=True
    )

    image = zoom_layer(
        image,
        training=True
    )

    return image


def preprocess_and_augment_image(
    image_path,
    label,
    rotation_layer,
    zoom_layer
):

    image = preprocess_image(image_path)

    image = augment_image(
        image,
        rotation_layer,
        zoom_layer
    )

    return image, label


# 15. Create Augmented Weighted Train Dataset

def create_augmented_weighted_train_dataset(
    train_df,
    y_train,
    class_weights,
    batch_size=BATCH_SIZE
):

    rotation_layer, zoom_layer = create_augmentation_layers()

    class_weights_tensor = tf.constant(
        [
            class_weights[i]
            for i in range(len(class_weights))
        ],
        dtype=tf.float32
    )

    train_dataset_augmented = tf.data.Dataset.from_tensor_slices(
        (train_df['image_path'].values, y_train)
    )

    train_dataset_augmented = train_dataset_augmented.shuffle(
        buffer_size=len(train_df),
        seed=42,
        reshuffle_each_iteration=True
    )

    train_dataset_augmented = train_dataset_augmented.map(
        lambda path, label: preprocess_and_augment_image(
            path,
            label,
            rotation_layer,
            zoom_layer
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset_augmented = train_dataset_augmented.batch(
        batch_size
    )

    train_dataset_augmented = train_dataset_augmented.map(
        lambda images, labels: add_sample_weights(
            images,
            labels,
            class_weights_tensor
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    train_dataset_augmented = train_dataset_augmented.prefetch(
        tf.data.AUTOTUNE
    )

    return train_dataset_augmented


# 16. Convert Image Paths to Relative Paths

def make_relative_paths(dataframe, dataset_path):

    result = dataframe.copy()

    result['image_path'] = result['image_path'].apply(
        lambda path: os.path.relpath(path, dataset_path)
    )

    return result


# 17. Save Processed Data and Metadata

def save_processed_data(
    train_df,
    val_df,
    test_df,
    label_encoder,
    class_weights,
    dataset_path,
    output_path,
    original_images,
    removed_duplicates,
    batch_size=BATCH_SIZE
):

    os.makedirs(output_path, exist_ok=True)

    train_export = make_relative_paths(
        train_df,
        dataset_path
    )

    val_export = make_relative_paths(
        val_df,
        dataset_path
    )

    test_export = make_relative_paths(
        test_df,
        dataset_path
    )

    # Save dataset splits

    train_export.to_csv(
        os.path.join(output_path, 'train.csv'),
        index=False
    )

    val_export.to_csv(
        os.path.join(output_path, 'validation.csv'),
        index=False
    )

    test_export.to_csv(
        os.path.join(output_path, 'test.csv'),
        index=False
    )

    # Save class names

    with open(
        os.path.join(output_path, 'label_classes.json'),
        'w',
        encoding='utf-8'
    ) as file:

        json.dump(
            label_encoder.classes_.tolist(),
            file,
            indent=4,
            ensure_ascii=False
        )

    # Save class weights

    weights_by_class = {
        class_name: float(class_weights[i])
        for i, class_name in enumerate(label_encoder.classes_)
    }

    with open(
        os.path.join(output_path, 'class_weights.json'),
        'w',
        encoding='utf-8'
    ) as file:

        json.dump(
            weights_by_class,
            file,
            indent=4,
            ensure_ascii=False
        )

    # Calculate actual dataset information

    clean_images = (
        len(train_df) +
        len(val_df) +
        len(test_df)
    )

    removed_duplicates_count = len(removed_duplicates)

    info = {
        'number_of_classes': len(label_encoder.classes_),
        'original_images': int(original_images),
        'clean_images': int(clean_images),
        'removed_duplicates': int(removed_duplicates_count),
        'train_images': len(train_df),
        'validation_images': len(val_df),
        'test_images': len(test_df),
        'image_size': [IMG_SIZE, IMG_SIZE],
        'normalization': '0-1',
        'batch_size': batch_size,
        'augmentation': [
            'RandomRotation',
            'RandomZoom'
        ]
    }

    # Save preprocessing information

    with open(
        os.path.join(output_path, 'preprocessing_info.json'),
        'w',
        encoding='utf-8'
    ) as file:

        json.dump(
            info,
            file,
            indent=4,
            ensure_ascii=False
        )

    return output_path
