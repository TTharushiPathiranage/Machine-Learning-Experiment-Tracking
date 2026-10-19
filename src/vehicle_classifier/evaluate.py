"""Evaluate a trained model on the held-out test split.

Recomputes the same stratified split used by train.py (same SEED, same data
root) rather than persisting test paths separately, so this must be run
against the same dataset the model was trained on.

Usage:
    python -m vehicle_classifier.evaluate --architecture efficientnetb0
    python -m vehicle_classifier.evaluate --architecture mobilenetv2
"""

import argparse
import json

import mlflow
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from tensorflow import keras

from vehicle_classifier.config import ARCHITECTURE_CHOICES, DATA_ROOT, DEFAULT_ARCHITECTURE, OUTPUTS_DIR, resolve_model_path
from vehicle_classifier.data import discover_classes, load_and_clean, make_dataset, stratified_split
from vehicle_classifier.tracking import start_run


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=ARCHITECTURE_CHOICES, default=DEFAULT_ARCHITECTURE)
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--model-path", default=None, help="override the default models/vehicle_classifier_<architecture>.keras")
    return p.parse_args()


def evaluate(architecture, data_root=DATA_ROOT, model_file=None):
    """Returns a results dict; also usable as a library call from compare.py."""
    model_file = model_file or resolve_model_path(architecture)
    class_names = discover_classes(data_root)
    df = load_and_clean(data_root, class_names=class_names, verbose=False)
    _, _, (p_test, y_test) = stratified_split(df)
    test_ds = make_dataset(p_test, y_test)

    keras.backend.clear_session()  # avoid graph/session buildup across repeated calls (e.g. from an API server)
    model = keras.models.load_model(model_file)
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    y_proba = model.predict(test_ds, verbose=0)
    y_pred = y_proba.argmax(axis=1)

    report = classification_report(
        y_test, y_pred, target_names=class_names, digits=3, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names))).tolist()
    try:
        macro_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro"))
    except ValueError:
        macro_auc = None

    return {
        "architecture": architecture,
        "model_path": str(model_file),
        "num_params": int(model.count_params()),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "confusion_matrix": cm,
        "classification_report": report,
        "macro_roc_auc": macro_auc,
        "class_names": class_names,
    }


def evaluate_and_log(architecture, data_root=DATA_ROOT, model_file=None):
    """Evaluate, add inference latency, and record everything as an MLflow evaluate run."""
    from vehicle_classifier.compare import measure_inference_latency  # lazy: compare imports this module

    with start_run(f"eval-{architecture}", stage="evaluate", params={"architecture": architecture, "data_root": str(data_root)}):
        results = evaluate(architecture, data_root, model_file)
        results["inference_ms_per_image"] = measure_inference_latency(architecture, data_root)
        mlflow.log_dict(results, "results.json")
        mlflow.log_metric("inference_ms_per_image", results["inference_ms_per_image"])
        mlflow.log_params({"model_path": results["model_path"], "num_params": results["num_params"]})
        mlflow.log_metrics({"test_loss": results["test_loss"], "test_accuracy": results["test_accuracy"]})
        if results["macro_roc_auc"] is not None:
            mlflow.log_metric("macro_roc_auc", results["macro_roc_auc"])
        mlflow.log_dict(results["classification_report"], "classification_report.json")
        mlflow.log_dict({"labels": results["class_names"], "matrix": results["confusion_matrix"]}, "confusion_matrix.json")
    return results


def main():
    args = parse_args()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    results = evaluate_and_log(args.architecture, args.data_root, args.model_path)

    print(f"Test loss     {results['test_loss']:.4f}")
    print(f"Test accuracy {results['test_accuracy']:.4f}")
    print(f"Params        {results['num_params']:,}")
    print(json.dumps(results["classification_report"], indent=2))

    out_path = OUTPUTS_DIR / f"evaluation_{args.architecture}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
