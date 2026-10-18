# Vehicle Classifier

Transfer-learning image classifier for three vehicle classes: `airplanes`,
`cars`, `ship`. Two backbone architectures are supported so they can be
trained and compared against each other:

- **efficientnetb0** (default) - EfficientNetB0, normalization baked into the model
- **mobilenetv2** - MobileNetV2, much smaller/faster, used as a comparison point

## Known data caveats

- Only 48 images total (16 per class). Treat single-split accuracy as noisy;
  prefer the 5-fold cross-validation number.
- All `ship` images come from a different source at much lower resolution
  than `airplanes`/`cars`. After resizing to 224x224 they carry visible
  upscaling blur the other two classes don't, so a model can learn
  "blurry means ship" instead of real vehicle features. Check
  `outputs/gradcam/` before trusting a high accuracy score.

## Project layout

```
.
├── app/
│   └── main.py                            # FastAPI service: predict, evaluate, compare
├── data/vehicles/{airplanes,cars,ship}/   # one subfolder per class
├── models/                                # vehicle_classifier_<arch>.keras, checkpoints, class_names.json
├── notebooks/                             # one exploratory notebook per architecture (EDA, plots)
├── outputs/                               # evaluation_<arch>.json, cross_validation_<arch>.json, comparison.json, gradcam/<arch>/
└── src/vehicle_classifier/
    ├── config.py          # paths, hyperparameters, per-architecture path helpers
    ├── data.py            # load/clean/split, tf.data pipeline
    ├── model.py           # backbone registry (EfficientNetB0 / MobileNetV2) + fine-tune helpers
    ├── train.py            # stage-1 head training + stage-2 fine-tuning
    ├── evaluate.py         # test-set metrics for one architecture
    ├── compare.py          # evaluates all trained architectures and prints a comparison table
    ├── cross_validate.py   # 5-fold stratified CV
    ├── gradcam.py          # Grad-CAM heatmaps
    └── predict.py          # VehicleClassifier - used by both the CLI and app/main.py
```

## Setup

Create the conda environment (only needed once):

```bash
conda create -n myml python=3.12
```

Activate it and install the project:

```bash
conda activate myml
pip install -r requirements.txt   # or requirements-dev.txt to also get Jupyter
pip install -e .
```

## Usage

Train one architecture at a time (`--architecture efficientnetb0` or `mobilenetv2`, default efficientnetb0):

```bash
python -m vehicle_classifier.train --architecture efficientnetb0
python -m vehicle_classifier.train --architecture mobilenetv2
```

Evaluate, cross-validate or run Grad-CAM for a single architecture the same way:

```bash
python -m vehicle_classifier.evaluate --architecture mobilenetv2
python -m vehicle_classifier.cross_validate --architecture mobilenetv2
python -m vehicle_classifier.gradcam --architecture mobilenetv2
```

Once both are trained, compare them side by side (accuracy, loss, macro ROC-AUC,
parameter count, inference latency), written to `outputs/comparison.json`:

```bash
python -m vehicle_classifier.compare
```

Predict a single image:

```bash
python -m vehicle_classifier.predict path/to/image.jpg --architecture efficientnetb0
```

## API

A FastAPI service wraps prediction and evaluation for both models, with
interactive Swagger docs:

```bash
uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000/docs. Endpoints:

| Endpoint | Method | What it does |
|---|---|---|
| `/predict/{architecture}` | POST | Upload an image, get back the predicted class, confidence and per-class probabilities for that model |
| `/evaluate/{architecture}` | GET | Test-set metrics for one model: accuracy, loss, confusion matrix, classification report, macro ROC-AUC, param count, inference latency |
| `/compare` | GET | Runs `/evaluate` for both models and returns them side by side, plus which one wins on accuracy and which wins on latency |

`{architecture}` is `efficientnetb0` or `mobilenetv2`; both endpoints 404 if
that architecture hasn't been trained yet (see Usage above). Loaded
classifiers are cached in memory across requests so repeated predictions
don't reload the model each time.

### Examples

Predict:

```bash
curl -X POST -F "file=@data/vehicles/cars/cars98.jpg" http://127.0.0.1:8000/predict/efficientnetb0
```

```json
{"architecture":"efficientnetb0","predicted_class":"cars","confidence":0.9988,
 "probabilities":{"airplanes":0.0006,"cars":0.9988,"ship":0.0005}}
```

Evaluate one model:

```bash
curl http://127.0.0.1:8000/evaluate/mobilenetv2
```

```json
{"architecture":"mobilenetv2","num_params":2266947,"test_loss":0.0039,
 "test_accuracy":1.0,"confusion_matrix":[[3,0,0],[0,3,0],[0,0,3]],
 "classification_report":{"...":"..."},"macro_roc_auc":1.0,
 "class_names":["airplanes","cars","ship"],"inference_ms_per_image":86.7}
```

Compare both:

```bash
curl http://127.0.0.1:8000/compare
```

```json
{"models":{"efficientnetb0":{"...":"..."},"mobilenetv2":{"...":"..."}},
 "best_by_test_accuracy":"efficientnetb0","best_by_inference_latency":"mobilenetv2"}
```

## Roadmap

- **Docker**: containerize training and the API so experiment runs and
  serving are reproducible (e.g. one image per experiment, mount `data/`
  and `outputs/` as volumes).

Deliberately not set up yet.
