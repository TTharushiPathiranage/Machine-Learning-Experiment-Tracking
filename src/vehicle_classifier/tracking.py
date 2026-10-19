"""MLflow experiment tracking helpers shared by train.py and evaluate.py."""

from contextlib import contextmanager

import mlflow
from mlflow import MlflowClient
from tensorflow import keras

from vehicle_classifier.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI


@contextmanager
def start_run(run_name, stage, params=None):
    """Open an MLflow run in the project experiment, tagged with its pipeline stage."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("stage", stage)
        if params:
            mlflow.log_params(params)
        yield run


class EpochLogger(keras.callbacks.Callback):
    """Logs every Keras epoch metric to the active MLflow run.

    step_offset lets stage-2 fine-tuning continue the x-axis after stage-1.
    """

    def __init__(self, prefix="", step_offset=0):
        super().__init__()
        self.prefix = prefix
        self.step_offset = step_offset

    def on_epoch_end(self, epoch, logs=None):
        metrics = {f"{self.prefix}{k}": float(v) for k, v in (logs or {}).items()}
        mlflow.log_metrics(metrics, step=self.step_offset + epoch)


def latest_evaluation(architecture):
    """Newest finished evaluate run's full results dict for an architecture, or None.

    Only runs that logged results.json and inference latency qualify, so runs
    from before that was tracked are skipped.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = MlflowClient().get_experiment_by_name(MLFLOW_EXPERIMENT)
    if experiment is None:
        return None
    runs = MlflowClient().search_runs(
        [experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' AND tags.stage = 'evaluate' "
            f"AND params.architecture = '{architecture}' AND metrics.inference_ms_per_image > 0"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return mlflow.artifacts.load_dict(f"runs:/{runs[0].info.run_id}/results.json")
