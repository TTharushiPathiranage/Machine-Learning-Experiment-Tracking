"""Grad-CAM: visualize what the model is attending to.

Given the resolution confound between classes flagged in the original EDA
(ship images are much lower-resolution than airplanes/cars), this checks
whether the model attends to the object itself or to background/compression
artifacts. Heat on the hull, wings or wheels is a good sign; heat spread
flatly over sky or water is a warning.

Usage:
    python -m vehicle_classifier.gradcam --architecture efficientnetb0
    python -m vehicle_classifier.gradcam --architecture mobilenetv2
"""

import argparse

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras

from vehicle_classifier.config import ARCHITECTURE_CHOICES, DATA_ROOT, DEFAULT_ARCHITECTURE, IMG_SIZE, OUTPUTS_DIR, model_path
from vehicle_classifier.data import discover_classes, load_and_clean, make_dataset, stratified_split
from vehicle_classifier.model import LAST_CONV_LAYER


def gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = keras.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heat = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heat = tf.maximum(heat, 0) / (tf.reduce_max(heat) + 1e-8)
    return heat.numpy()


def save_overlay(raw_uint8, heat, out_path):
    heat_rs = np.array(Image.fromarray(np.uint8(255 * heat)).resize((IMG_SIZE, IMG_SIZE)))
    heat_rgb = np.asarray(Image.fromarray(heat_rs).convert("L").convert("RGB"))
    overlay = (0.6 * raw_uint8 + 0.4 * heat_rgb).astype("uint8")
    Image.fromarray(overlay).save(out_path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--architecture", choices=ARCHITECTURE_CHOICES, default=DEFAULT_ARCHITECTURE)
    p.add_argument("--data-root", default=str(DATA_ROOT))
    p.add_argument("--model-path", default=None)
    p.add_argument("--n-images", type=int, default=6)
    return p.parse_args()


def main():
    args = parse_args()
    model_file = args.model_path or model_path(args.architecture)
    out_dir = OUTPUTS_DIR / "gradcam" / args.architecture
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = discover_classes(args.data_root)
    df = load_and_clean(args.data_root, class_names=class_names, verbose=False)
    _, _, (p_test, y_test) = stratified_split(df)
    test_ds = make_dataset(p_test, y_test)

    model = keras.models.load_model(model_file)
    last_conv_layer_name = LAST_CONV_LAYER[args.architecture]
    test_imgs = np.concatenate([x.numpy() for x, _ in test_ds])
    y_proba = model.predict(test_ds, verbose=0)
    y_pred = y_proba.argmax(axis=1)

    n_show = min(args.n_images, len(test_imgs))
    for i in range(n_show):
        raw = test_imgs[i].astype("uint8")
        heat = gradcam_heatmap(raw[None, ...].astype("float32"), model, last_conv_layer_name)
        ok = "ok" if y_pred[i] == y_test[i] else "WRONG"
        out_path = out_dir / f"{i:02d}_{class_names[y_test[i]]}_pred-{class_names[y_pred[i]]}_{ok}.png"
        save_overlay(raw, heat, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
