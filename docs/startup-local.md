# Training Pipeline (Local)

## 1) Setup
1. Create and activate a virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 2) Data (datasets)
`scripts/local/train_models.py` expects a dataset directory with a `data.yaml` in the repo root, e.g.:
- `data_640/data.yaml`
- `data_420/data.yaml`
- `fruit_dataset_640/data.yaml`
- `data_640_demo_day/data.yaml`

Note: dataset directories are typically managed externally (see `docs/artifact-policy.md`).
The repo expects the directory names above to exist on disk.

## 3) Pretrained weights
`scripts/local/train_models.py` lädt die Basismodell-Weights aus:
- `yolo_weights/<base_model>.pt`

For example:
- `yolo_weights/yolo11n-seg.pt`
- `yolo_weights/yolo26n-seg.pt`

If the file is missing, the script fails with a download instruction.
Create the `yolo_weights/` folder locally and download the needed `.pt` files.

## 4) Train
Train YOLO segmentation models and run the augmentation experiments:
```bash
python3 scripts/local/train_models.py --model yolo11n-seg --dataset data_640 --augmentation all
```

CLI options:
- `--model`: `yolo11n-seg` or `yolo26n-seg`
- `--dataset`: `data_420`, `data_640`, `fruit_dataset_640`, `data_640_demo_day`
- `--augmentation`: `no_augmentation`, `baseline`, `moderate_geom`, `strong_geom`, `strong_geom_fruit`, or `all`

## 5) Compare / Evaluate
To compare all trained models for the hard-coded timestamps used by the repo:
```bash
python3 scripts/local/compare_models.py
```

For the demo-day dataset:
```bash
python3 scripts/local/compare_models_demo_day.py
```

Important: `compare_models*.py` uses a `MODEL_TIMESTAMPS` mapping. If you trained new runs after the last recorded timestamp(s), you may need to update that mapping so the script can find the correct `runs/.../weights/best.pt` files.

