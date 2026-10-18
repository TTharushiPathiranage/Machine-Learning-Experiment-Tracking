"""Transfer-learning model builders for every supported backbone architecture.

Each backbone has different input-normalization expectations:
  - EfficientNetB0 has rescaling/normalization baked into its graph and wants
    raw 0-255 pixels. Adding an external Rescaling layer in front of it is
    the single most common mistake with this backbone.
  - MobileNetV2 expects pixels scaled to [-1, 1]. We add that scaling as a
    Rescaling layer inside the model graph (not in the data pipeline), so the
    same decode()/make_dataset() pipeline in data.py works for every
    architecture unmodified, and the saved .keras file is self-contained.
"""

from tensorflow import keras
from tensorflow.keras import layers

from vehicle_classifier.config import IMG_SIZE

# last conv/activation layer name per backbone, used by gradcam.py
LAST_CONV_LAYER = {
    "efficientnetb0": "top_conv",
    "mobilenetv2": "out_relu",
}

_BACKBONE_BUILDERS = {
    "efficientnetb0": keras.applications.EfficientNetB0,
    "mobilenetv2": keras.applications.MobileNetV2,
}


def build_augmentation():
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.10),
            layers.RandomZoom(0.15),
            layers.RandomTranslation(0.10, 0.10),
            layers.RandomContrast(0.15),
        ],
        name="data_augmentation",
    )


def build_model(architecture, num_classes, img_size=IMG_SIZE, dropout=0.3, learning_rate=1e-3):
    """Stage-1 model: frozen backbone + a small trainable head.

    Returns (model, base_model) so callers can later unfreeze base_model's top
    layers for stage-2 fine-tuning.
    """
    if architecture not in _BACKBONE_BUILDERS:
        raise ValueError(f"unknown architecture {architecture!r}, choose from {list(_BACKBONE_BUILDERS)}")

    data_augmentation = build_augmentation()

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    if architecture == "mobilenetv2":
        x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescale_to_pm1")(x)

    base = _BACKBONE_BUILDERS[architecture](
        include_top=False,
        weights="imagenet",
        input_tensor=x,
    )
    base.trainable = False

    x = layers.GlobalAveragePooling2D(name="gap")(base.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs, outputs, name=f"{architecture}_vehicles")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def unfreeze_for_finetuning(model, base_model, num_unfrozen_layers=20, learning_rate=1e-5):
    """Stage-2: unfreeze the backbone's last `num_unfrozen_layers`, keep BatchNorm frozen.

    BatchNorm stays frozen on purpose: with a small batch size, recomputing
    batch statistics destabilises the pretrained features.
    """
    base_model.trainable = True
    for layer in base_model.layers[:-num_unfrozen_layers]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
