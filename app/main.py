"""FastAPI service for the vehicle classifier.

Exposes per-architecture prediction and evaluation, plus a head-to-head
comparison endpoint, over the two trained models (efficientnetb0, mobilenetv2).

Run:
    uvicorn app.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

import io
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

from vehicle_classifier.compare import measure_inference_latency
from vehicle_classifier.config import ARCHITECTURE_CHOICES, DATA_ROOT, resolve_model_path
from vehicle_classifier.evaluate import evaluate as evaluate_architecture
from vehicle_classifier.predict import VehicleClassifier
from vehicle_classifier.tracking import latest_evaluation


class Architecture(str, Enum):
    efficientnetb0 = "efficientnetb0"
    mobilenetv2 = "mobilenetv2"


class PredictionResponse(BaseModel):
    architecture: str
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]


class EvaluationResponse(BaseModel):
    architecture: str
    model_path: str
    num_params: int
    test_loss: float
    test_accuracy: float
    confusion_matrix: List[List[int]]
    classification_report: Dict[str, Any]
    macro_roc_auc: Optional[float]
    class_names: List[str]
    inference_ms_per_image: float


class ComparisonResponse(BaseModel):
    models: Dict[str, EvaluationResponse]
    best_by_test_accuracy: str
    best_by_inference_latency: str


app = FastAPI(
    title="Vehicle Classifier API",
    description=(
        "Predict vehicle class (airplanes/cars/ship) and inspect or compare "
        "evaluation metrics for the EfficientNetB0 and MobileNetV2 models."
    ),
    version="0.1.0",
)


def _require_trained(architecture: str) -> None:
    if not resolve_model_path(architecture).exists():
        raise HTTPException(status_code=404, detail=f"no trained model found for architecture={architecture!r}")


@lru_cache(maxsize=len(ARCHITECTURE_CHOICES))
def _get_classifier(architecture: str) -> VehicleClassifier:
    _require_trained(architecture)
    return VehicleClassifier(architecture=architecture)


def _evaluate_with_latency(architecture: str) -> dict:
    results = evaluate_architecture(architecture, DATA_ROOT)
    results["inference_ms_per_image"] = measure_inference_latency(architecture, DATA_ROOT)
    return results


@app.get("/", include_in_schema=False)
def root():
    return {"docs": "/docs"}


@app.post("/predict/{architecture}", response_model=PredictionResponse, tags=["predict"])
async def predict(architecture: Architecture, file: UploadFile = File(...)):
    """Predict the vehicle class of an uploaded image using the given model."""
    clf = _get_classifier(architecture.value)

    image_bytes = await file.read()
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception:
        raise HTTPException(status_code=400, detail="uploaded file is not a readable image")

    label, confidence = clf.predict(image)
    probabilities = clf.predict_proba(image)
    return PredictionResponse(
        architecture=architecture.value,
        predicted_class=label,
        confidence=confidence,
        probabilities=probabilities,
    )


@app.get("/evaluate/{architecture}", response_model=EvaluationResponse, tags=["evaluate"])
def get_evaluation(architecture: Architecture):
    """Evaluate one model on the held-out test split and return its metrics."""
    _require_trained(architecture.value)
    return _evaluate_with_latency(architecture.value)


@app.get("/compare", response_model=ComparisonResponse, tags=["evaluate"])
def compare_models():
    """Compare both models head-to-head using each one's latest MLflow evaluation run.

    Produce those runs with `python -m vehicle_classifier.evaluate --architecture <arch>`.
    """
    results = {arch: latest_evaluation(arch) for arch in ARCHITECTURE_CHOICES}
    missing = [a for a, r in results.items() if r is None]
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"no MLflow evaluation run for: {missing}; run `python -m vehicle_classifier.evaluate --architecture <arch>` first",
        )

    best_by_accuracy = max(results, key=lambda a: results[a]["test_accuracy"])
    best_by_latency = min(results, key=lambda a: results[a]["inference_ms_per_image"])

    return ComparisonResponse(
        models=results,
        best_by_test_accuracy=best_by_accuracy,
        best_by_inference_latency=best_by_latency,
    )
