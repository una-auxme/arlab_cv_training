# Validation (after the structure change)

Goal: verify that the new folder structure for Local (`scripts/local/`) and SLURM (`scripts/slurm/`) correctly matches the workflow, without breaking the training/evaluation logic.

## 1) Local (no GPU / no training)
1. Syntax check of the Python entry points:
```bash
python3 -m py_compile scripts/local/train_models.py scripts/local/compare_models.py scripts/local/compare_models_demo_day.py
```

2. (optional) CLI check in the active venv (imports `ultralytics`):
```bash
source .venv/bin/activate
python3 scripts/local/train_models.py --help
```

## 2) SLURM (no submit)
1. Shell syntax check:
```bash
bash -n scripts/slurm/*.sh
```

2. Path check:
- The SLURM scripts `cd` into the repo root (`.../arlab_cv_training`)
- The `srun python3 ...` calls point to `scripts/local/*.py`

## 3) SLURM submit (only if prerequisites are satisfied)
Examples:
```bash
sbatch scripts/slurm/slurm_train_yolo11.sh
sbatch scripts/slurm/slurm_compare_models.sh
```

Prerequisites:
- `yolo_weights/yolo11n-seg.pt` and `yolo_weights/yolo26n-seg.pt` exist
- Dataset folders (including `data.yaml`) exist on disk (e.g. `data_640/data.yaml`)

