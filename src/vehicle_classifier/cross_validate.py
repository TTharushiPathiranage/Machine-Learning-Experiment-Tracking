"""5-fold stratified cross-validation.

With only 9 images in a single test split, one misclassification swings
accuracy by ~11 points. This reuses every image as test data exactly once
and reports mean +/- std, which is the number worth quoting.

Usage:
    python -m vehicle_classifier.cross_validate --architecture efficientnetb0
    python -m vehicle_classifier.cross_validate --architecture mobilenetv2
"""

import argparse
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

from vehicle_classifier.config import ARCHITECTURE_CHOICES, DATA_ROOT, DEFAULT_ARCHITECTURE, OUTPUTS_DIR, SEED
from vehicle_classifier.data import discover_classes, load_and_clean, make_dataset
from vehicle_classifier.model import build_model
from vehicle_classifier.train import set_seeds


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=ARCHITECTURE_CHOICES, default=DEFAULT_ARCHITECTURE)
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--epochs", type=int, default=25)
    return p.parse_args()


def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    class_names = discover_classes(args.data_root)
    df = load_and_clean(args.data_root, class_names=class_names, verbose=False)
    paths = df["path"].values
    labels = df["label"].values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=SEED)
    fold_scores = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(paths, labels), start=1):
        keras.backend.clear_session()
        set_seeds()
        model, _ = build_model(args.architecture, num_classes=len(class_names))
        model.fit(make_dataset(paths[tr_idx], labels[tr_idx], shuffle=True), epochs=args.epochs, verbose=0)
        acc = model.evaluate(make_dataset(paths[te_idx], labels[te_idx]), verbose=0)[1]
        fold_scores.append(acc)
        print(f"fold {fold}: accuracy {acc:.3f}  (n_test={len(te_idx)})")

    mean, std = float(np.mean(fold_scores)), float(np.std(fold_scores))
    print(f"\n{args.n_splits}-fold CV accuracy: {mean:.3f} +/- {std:.3f}")

    out_path = OUTPUTS_DIR / f"cross_validation_{args.architecture}.json"
    with open(out_path, "w") as f:
        json.dump({"architecture": args.architecture, "fold_scores": fold_scores, "mean": mean, "std": std}, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
