"""ZenML pipelines for the vehicle classifier.

Usage:
    python -m pipelines.run --pipeline evaluation --architecture mobilenetv2
    python -m pipelines.run --pipeline training --architecture mobilenetv2 --stage1-epochs 40 --stage2-epochs 20
"""

import argparse

from zenml import pipeline

from pipelines.steps import current_model, evaluate_model, train_model
from vehicle_classifier.config import ARCHITECTURE_CHOICES, DEFAULT_ARCHITECTURE


@pipeline
def training_pipeline(architecture: str, stage1_epochs: int = 40, stage2_epochs: int = 20, unfreeze_layers: int = 20):
    """Train a model, then evaluate it."""
    model_file = train_model(architecture, stage1_epochs, stage2_epochs, unfreeze_layers)
    evaluate_model(architecture, model_file)


@pipeline
def evaluation_pipeline(architecture: str):
    """Evaluate the existing trained model without retraining."""
    model_file = current_model(architecture)
    evaluate_model(architecture, model_file)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pipeline", choices=["training", "evaluation"], default="evaluation")
    p.add_argument("--architecture", choices=ARCHITECTURE_CHOICES, default=DEFAULT_ARCHITECTURE)
    p.add_argument("--stage1-epochs", type=int, default=40)
    p.add_argument("--stage2-epochs", type=int, default=20)
    p.add_argument("--unfreeze-layers", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    if args.pipeline == "training":
        training_pipeline(args.architecture, args.stage1_epochs, args.stage2_epochs, args.unfreeze_layers)
    else:
        evaluation_pipeline(args.architecture)


if __name__ == "__main__":
    main()
