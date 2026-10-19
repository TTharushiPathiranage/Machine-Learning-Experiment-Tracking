"""ZenML steps wrapping the vehicle_classifier train and evaluate code.

MLflow tracking still happens inside the wrapped functions, so ZenML adds
orchestration, caching and artifact lineage on top of it.
"""

from argparse import Namespace

from zenml import step

from vehicle_classifier.config import DATA_ROOT, OUTPUTS_DIR, resolve_model_path
from vehicle_classifier.evaluate import evaluate_and_log
from vehicle_classifier.tracking import start_run
from vehicle_classifier.train import train


@step(enable_cache=False)  # inputs are names/paths, not file contents, so cache hits would hide retrains
def train_model(architecture: str, stage1_epochs: int = 40, stage2_epochs: int = 20, unfreeze_layers: int = 20) -> str:
    """Train both stages and return the path of the saved model."""
    args = Namespace(
        architecture=architecture,
        data_root=str(DATA_ROOT),
        stage1_epochs=stage1_epochs,
        stage2_epochs=stage2_epochs,
        unfreeze_layers=unfreeze_layers,
    )
    with start_run(architecture, stage="train", params=vars(args)):
        train(args)
    return str(resolve_model_path(architecture))


@step(enable_cache=False)  # inputs are names/paths, not file contents, so cache hits would hide retrains
def evaluate_model(architecture: str, model_file: str) -> dict:
    """Evaluate a saved model on the held-out test split and log the run to MLflow."""
    import json

    results = evaluate_and_log(architecture, DATA_ROOT, model_file)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUTS_DIR / f"evaluation_{architecture}.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


@step(enable_cache=False)  # inputs are names/paths, not file contents, so cache hits would hide retrains
def current_model(architecture: str) -> str:
    """Resolve the already-trained model file, for evaluation-only runs."""
    return str(resolve_model_path(architecture))
