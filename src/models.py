"""Baseline models for PlantVillage leaf disease classification.

Includes:
1. Trivial Baseline (majority-class predictor via DummyClassifier).
2. Simple Learned Baseline (32x32 flattened RGB pixels + StandardScaler + LogisticRegression).

Primary metric: Macro-F1 (alongside Accuracy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from PIL import Image
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_images_from_dir(
    root_dir: Path,
    image_size: Tuple[int, int] = (32, 32),
    max_samples: int | None = None,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """Load RGB images organized as root_dir/<class_name>/<image_file>."""
    if not root_dir.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    class_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
    label_to_idx = {d.name: idx for idx, d in enumerate(class_dirs)}
    class_names = [d.name for d in class_dirs]

    records: list[Tuple[Path, int]] = []
    for class_dir in class_dirs:
        label_idx = label_to_idx[class_dir.name]
        for img_path in sorted(class_dir.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTS:
                records.append((img_path, label_idx))

    if not records:
        raise ValueError(f"No valid images found under: {root_dir}")

    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    if max_samples is not None:
        indices = indices[: min(max_samples, len(indices))]

    features: list[np.ndarray] = []
    targets: list[int] = []
    for idx in indices:
        img_path, label_idx = records[int(idx)]
        with Image.open(img_path) as img:
            rgb_img = img.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
            arr = np.asarray(rgb_img, dtype=np.float32).reshape(-1) / 255.0
            features.append(arr)
            targets.append(label_idx)

    return np.vstack(features), np.asarray(targets, dtype=np.int64), class_names


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute Accuracy and Macro-F1."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def run_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int = 42,
    max_iter: int = 500,
) -> dict[str, dict[str, float]]:
    """Fit and evaluate Trivial and Simple Learned baselines."""
    trivial_clf = DummyClassifier(strategy="most_frequent", random_state=seed)
    trivial_clf.fit(x_train, y_train)
    trivial_metrics = evaluate_predictions(y_eval, trivial_clf.predict(x_eval))

    logreg_clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, random_state=seed),
    )
    logreg_clf.fit(x_train, y_train)
    logreg_metrics = evaluate_predictions(y_eval, logreg_clf.predict(x_eval))

    return {
        "trivial_most_frequent": trivial_metrics,
        "logreg_32x32_rgb": logreg_metrics,
    }


def generate_synthetic_split(
    n_samples: int,
    n_features: int = 32 * 32 * 3,
    n_classes: int = 39,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic features/labels solely for smoke-testing the pipeline."""
    rng = np.random.default_rng(seed)
    x = rng.random((n_samples, n_features), dtype=np.float32)
    # Slightly skewed class distribution to mimic imbalance
    probs = np.exp(-np.linspace(0, 2.0, n_classes))
    probs /= probs.sum()
    y = rng.choice(n_classes, size=n_samples, p=probs).astype(np.int64)
    return x, y


def format_report(split_name: str, metrics_by_model: dict[str, dict[str, float]]) -> str:
    lines = [f"=== Baseline Results ({split_name}) ==="]
    for model_name, m in metrics_by_model.items():
        lines.append(
            f"- {model_name}: Accuracy = {m['accuracy']:.4f} | Macro-F1 = {m['macro_f1']:.4f}"
        )
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PlantVillage baseline models.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("data/train"),
        help="Path to training split directory (class subfolders).",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("data/val"),
        help="Path to validation split directory (class subfolders).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=None,
        help="Optional path to held-out test split directory (use only for final eval).",
    )
    parser.add_argument("--img-size", type=int, default=32, help="Resize width/height (default: 32).")
    parser.add_argument("--max-train", type=int, default=3000, help="Max train samples for fast baseline.")
    parser.add_argument("--max-eval", type=int, default=1000, help="Max val/test samples for fast baseline.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max-iter", type=int, default=500, help="LogisticRegression max_iter.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("results/baseline_metrics.json"),
        help="Where to save metrics JSON.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic data when local dataset folders are not present.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    size = (args.img_size, args.img_size)

    if args.demo:
        x_train, y_train = generate_synthetic_split(args.max_train, args.img_size * args.img_size * 3, seed=args.seed)
        x_val, y_val = generate_synthetic_split(args.max_eval, args.img_size * args.img_size * 3, seed=args.seed + 1)
        val_results = run_baselines(x_train, y_train, x_val, y_val, seed=args.seed, max_iter=args.max_iter)
        print(format_report("val [DEMO]", val_results))
        all_results = {"val_demo": val_results}
    else:
        x_train, y_train, _ = load_images_from_dir(
            args.train_dir, image_size=size, max_samples=args.max_train, seed=args.seed
        )
        x_val, y_val, _ = load_images_from_dir(
            args.val_dir, image_size=size, max_samples=args.max_eval, seed=args.seed + 1
        )
        val_results = run_baselines(x_train, y_train, x_val, y_val, seed=args.seed, max_iter=args.max_iter)
        print(format_report("val", val_results))
        all_results = {"val": val_results}

        if args.test_dir is not None:
            x_test, y_test, _ = load_images_from_dir(
                args.test_dir, image_size=size, max_samples=args.max_eval, seed=args.seed + 2
            )
            test_results = run_baselines(x_train, y_train, x_test, y_test, seed=args.seed, max_iter=args.max_iter)
            print(format_report("test", test_results))
            all_results["test"] = test_results

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nSaved metrics to: {args.output_json}")


if __name__ == "__main__":
    main()
