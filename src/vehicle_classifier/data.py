"""Loading, cleaning, splitting and tf.data pipelines for the vehicles dataset."""

import glob
import hashlib
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split

from vehicle_classifier.config import (
    BATCH,
    DATA_ROOT,
    IMG_SIZE,
    SEED,
    TEST_VAL_FRACTION,
    VAL_TEST_SPLIT,
)


def discover_classes(data_root=DATA_ROOT):
    """Return sorted class names, one per subfolder of data_root."""
    return sorted(
        d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))
    )


def load_and_clean(data_root=DATA_ROOT, class_names=None, verbose=True):
    """Walk the dataset, dropping corrupt/duplicate/tiny files, and return a metadata DataFrame.

    Four checks, each of which silently poisons training if skipped:
    1. Corrupt/truncated files - PIL.verify() catches these.
    2. Exact duplicates - MD5 hash; a duplicate split across train/test leaks the label.
    3. Non-RGB modes - everything is converted to RGB.
    4. Degenerate images - anything smaller than 32px on a side is dropped.
    """
    if class_names is None:
        class_names = discover_classes(data_root)

    records = []
    seen_hashes = {}
    n_corrupt = n_dup = n_tiny = 0

    for ci, cls in enumerate(class_names):
        for path in sorted(glob.glob(os.path.join(data_root, cls, "*"))):
            try:
                Image.open(path).verify()
                im = Image.open(path)
                w, h = im.size
            except Exception as e:
                if verbose:
                    print(f"  corrupt, dropped: {path} ({e})")
                n_corrupt += 1
                continue

            if w < 32 or h < 32:
                if verbose:
                    print(f"  too small, dropped: {path} ({w}x{h})")
                n_tiny += 1
                continue

            digest = hashlib.md5(open(path, "rb").read()).hexdigest()
            if digest in seen_hashes:
                if verbose:
                    print(f"  duplicate of {os.path.basename(seen_hashes[digest])}, dropped: {path}")
                n_dup += 1
                continue
            seen_hashes[digest] = path

            arr = np.asarray(im.convert("RGB"))
            records.append(
                dict(
                    path=path,
                    cls=cls,
                    label=ci,
                    width=w,
                    height=h,
                    mode=im.mode,
                    aspect=w / h,
                    kb=os.path.getsize(path) / 1024,
                    mean_px=float(arr.mean()),
                    std_px=float(arr.std()),
                )
            )

    df = pd.DataFrame(records)
    if verbose:
        print(f"kept {len(df)} images | dropped: {n_corrupt} corrupt, {n_dup} duplicate, {n_tiny} tiny")
    return df


def stratified_split(df, seed=SEED):
    """Split a cleaned DataFrame into stratified train/val/test path and label arrays."""
    paths = df["path"].values
    labels = df["label"].values

    p_train, p_tmp, y_train, y_tmp = train_test_split(
        paths, labels, test_size=TEST_VAL_FRACTION, stratify=labels, random_state=seed
    )
    p_val, p_test, y_val, y_test = train_test_split(
        p_tmp, y_tmp, test_size=VAL_TEST_SPLIT, stratify=y_tmp, random_state=seed
    )
    return (p_train, y_train), (p_val, y_val), (p_test, y_test)


def decode(path, label):
    img = tf.io.read_file(path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)  # stays in 0-255 for EfficientNet
    return img, label


def make_dataset(paths, labels, batch=BATCH, shuffle=False, seed=SEED):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch).prefetch(tf.data.AUTOTUNE)
