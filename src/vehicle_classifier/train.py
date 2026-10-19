"""Train the vehicle classifier: stage-1 head training + stage-2 fine-tuning.

Usage:
    python -m vehicle_classifier.train --architecture efficientnetb0
    python -m vehicle_classifier.train --architecture mobilenetv2
"""

import argparse
import json
import random

import mlflow
import numpy as np
import tensorflow as tf
from tensorflow import keras

from vehicle_classifier.config import (
    ARCHITECTURE_CHOICES,
    CLASS_NAMES_PATH,
    DATA_ROOT,
    DEFAULT_ARCHITECTURE,
    MODELS_DIR,
    SEED,
    best_finetuned_path,
    best_head_path,
    model_path,
)
from vehicle_classifier.data import discover_classes, load_and_clean, make_dataset, stratified_split
from vehicle_classifier.model import build_model, unfreeze_for_finetuning
from vehicle_classifier.tracking import EpochLogger, start_run


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=ARCHITECTURE_CHOICES, default=DEFAULT_ARCHITECTURE)
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--stage1-epochs", type=int, default=40)
    p.add_argument("--stage2-epochs", type=int, default=20)
    p.add_argument("--unfreeze-layers", type=int, default=20)
    return p.parse_args()


def train(args):
    set_seeds()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = discover_classes(args.data_root)
    print("architecture:", args.architecture)
    print("classes:", class_names)

    df = load_and_clean(args.data_root, class_names=class_names)
    (p_train, y_train), (p_val, y_val), (p_test, y_test) = stratified_split(df)
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name:6s} n={len(y):3d}  per class: "
              f"{dict(zip(class_names, np.bincount(y, minlength=len(class_names))))}")

    train_ds = make_dataset(p_train, y_train, shuffle=True)
    val_ds = make_dataset(p_val, y_val)

    model, base_model = build_model(args.architecture, num_classes=len(class_names))
    print(f"trainable params (stage 1): {sum(np.prod(w.shape) for w in model.trainable_weights):,}")

    stage1_callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(str(best_head_path(args.architecture)), monitor="val_accuracy", save_best_only=True),
        EpochLogger(prefix="stage1_"),
    ]
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=args.stage1_epochs, callbacks=stage1_callbacks, verbose=1)

    unfreeze_for_finetuning(model, base_model, num_unfrozen_layers=args.unfreeze_layers)
    print(f"trainable params (stage 2): {sum(np.prod(w.shape) for w in model.trainable_weights):,}")

    stage2_callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(best_finetuned_path(args.architecture)), monitor="val_accuracy", save_best_only=True),
        EpochLogger(prefix="stage2_", step_offset=len(history1.epoch)),
    ]
    history2 = model.fit(train_ds, validation_data=val_ds, epochs=args.stage2_epochs, callbacks=stage2_callbacks, verbose=1)
    mlflow.log_metrics({
        "best_stage1_val_accuracy": max(history1.history["val_accuracy"]),
        "best_stage2_val_accuracy": max(history2.history["val_accuracy"]),
    })

    out_path = model_path(args.architecture)
    model.save(out_path)
    with open(CLASS_NAMES_PATH, "w") as f:
        json.dump(class_names, f)
    print(f"saved {out_path} and {CLASS_NAMES_PATH}")
    mlflow.log_artifact(str(out_path), artifact_path="model")
    mlflow.log_artifact(str(CLASS_NAMES_PATH), artifact_path="model")


def main():
    args = parse_args()
    with start_run(args.architecture, stage="train", params=vars(args)):
        train(args)


if __name__ == "__main__":
    main()
