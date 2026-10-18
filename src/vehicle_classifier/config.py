"""Central paths and hyperparameters shared by every script in this package."""

from pathlib import Path

SEED = 42
IMG_SIZE = 224  # native input size for both supported backbones
BATCH = 8

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "vehicles"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

DEFAULT_ARCHITECTURE = "efficientnetb0"
ARCHITECTURE_CHOICES = ("efficientnetb0", "mobilenetv2")


def model_path(architecture=DEFAULT_ARCHITECTURE):
    return MODELS_DIR / f"vehicle_classifier_{architecture}.keras"


def best_head_path(architecture=DEFAULT_ARCHITECTURE):
    return MODELS_DIR / f"best_head_{architecture}.keras"


def best_finetuned_path(architecture=DEFAULT_ARCHITECTURE):
    return MODELS_DIR / f"best_finetuned_{architecture}.keras"


# train/val/test split ratios, applied as in the original notebook:
# 62.5% train, then the remaining 37.5% split evenly into val/test
TEST_VAL_FRACTION = 0.375
VAL_TEST_SPLIT = 0.5
