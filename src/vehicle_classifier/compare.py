"""Compare two or more trained architectures on the same held-out test split.

Usage:
    python -m vehicle_classifier.compare
    python -m vehicle_classifier.compare --architectures efficientnetb0 mobilenetv2
"""

import argparse
import json
import time

import pandas as pd

from vehicle_classifier.config import ARCHITECTURE_CHOICES, DATA_ROOT, OUTPUTS_DIR, resolve_model_path
from vehicle_classifier.data import discover_classes, load_and_clean, make_dataset, stratified_split
from vehicle_classifier.evaluate import evaluate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architectures", nargs="+", choices=ARCHITECTURE_CHOICES, default=list(ARCHITECTURE_CHOICES))
    p.add_argument("--data-root", default=str(DATA_ROOT))
    return p.parse_args()


def measure_inference_latency(architecture, data_root, n_repeats=20):
    """Mean single-image predict latency in milliseconds, after a warm-up call."""
    from tensorflow import keras

    class_names = discover_classes(data_root)
    df = load_and_clean(data_root, class_names=class_names, verbose=False)
    _, _, (p_test, _) = stratified_split(df)
    sample_ds = make_dataset(p_test[:1], [0])
    x, _ = next(iter(sample_ds))

    model = keras.models.load_model(resolve_model_path(architecture))
    model.predict(x, verbose=0)  # warm-up

    start = time.perf_counter()
    for _ in range(n_repeats):
        model.predict(x, verbose=0)
    elapsed_ms = (time.perf_counter() - start) / n_repeats * 1000
    return elapsed_ms


def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    available = [a for a in args.architectures if resolve_model_path(a).exists()]
    missing = sorted(set(args.architectures) - set(available))
    if missing:
        print(f"skipping (no trained model found): {missing}")
    if not available:
        raise SystemExit("no trained models found; run train.py for at least one --architecture first")

    rows = []
    all_results = {}
    for arch in available:
        print(f"evaluating {arch}...")
        results = evaluate(arch, args.data_root)
        results["inference_ms_per_image"] = measure_inference_latency(arch, args.data_root)
        all_results[arch] = results
        rows.append(
            {
                "architecture": arch,
                "params": results["num_params"],
                "test_accuracy": round(results["test_accuracy"], 4),
                "test_loss": round(results["test_loss"], 4),
                "macro_roc_auc": round(results["macro_roc_auc"], 4) if results["macro_roc_auc"] is not None else None,
                "inference_ms_per_image": round(results["inference_ms_per_image"], 2),
            }
        )

    table = pd.DataFrame(rows).set_index("architecture")
    print("\n" + table.to_string())

    out_path = OUTPUTS_DIR / "comparison.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
