"""Single-image inference.

This is the module a future FastAPI service should import directly:
    from vehicle_classifier.predict import VehicleClassifier
    clf = VehicleClassifier()  # defaults to the efficientnetb0 model
    label, confidence = clf.predict(image_path_or_pil_image)

Usage (CLI):
    python -m vehicle_classifier.predict path/to/image.jpg --architecture efficientnetb0
"""

import argparse
import json

import numpy as np
from PIL import Image
from tensorflow import keras

from vehicle_classifier.config import ARCHITECTURE_CHOICES, CLASS_NAMES_PATH, DEFAULT_ARCHITECTURE, IMG_SIZE, model_path


class VehicleClassifier:
    """Loads a trained model once; reuse the instance across predictions."""

    def __init__(self, architecture=DEFAULT_ARCHITECTURE, model_file=None, class_names_path=CLASS_NAMES_PATH):
        self.architecture = architecture
        self.model = keras.models.load_model(model_file or model_path(architecture))
        with open(class_names_path) as f:
            self.class_names = json.load(f)

    def predict(self, image):
        """image: a file path, or a PIL.Image already in RGB."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype="float32")[None, ...]  # 0-255, no rescaling (baked into the model)
        proba = self.model.predict(arr, verbose=0)[0]
        idx = int(proba.argmax())
        return self.class_names[idx], float(proba[idx])

    def predict_proba(self, image):
        """Returns {class_name: probability} for every class."""
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        arr = np.asarray(img, dtype="float32")[None, ...]
        proba = self.model.predict(arr, verbose=0)[0]
        return {c: float(p) for c, p in zip(self.class_names, proba)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image_path")
    p.add_argument("--architecture", choices=ARCHITECTURE_CHOICES, default=DEFAULT_ARCHITECTURE)
    args = p.parse_args()

    clf = VehicleClassifier(architecture=args.architecture)
    label, confidence = clf.predict(args.image_path)
    print(f"{label} ({confidence:.1%})")
    for c, prob in sorted(clf.predict_proba(args.image_path).items(), key=lambda t: -t[1]):
        print(f"  {c:12s} {prob:6.2%}")


if __name__ == "__main__":
    main()
