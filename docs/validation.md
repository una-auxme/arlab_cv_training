# Validation (nach Strukturwechsel)

Ziel: sicherstellen, dass die neue Ordnerstruktur Local (`scripts/local/`) und SLURM (`scripts/slurm/`) korrekt abbildet, ohne die Trainings-/Evaluation-Logik zu brechen.

## 1) Lokal (ohne GPU/Training)
1. Syntaxcheck der Python Entrypoints:
```bash
python3 -m py_compile scripts/local/train_models.py scripts/local/compare_models.py scripts/local/compare_models_demo_day.py
```

2. (optional) CLI-Check im aktiven venv (importiert `ultralytics`):
```bash
source .venv/bin/activate
python3 scripts/local/train_models.py --help
```

## 2) SLURM (ohne Submit)
1. Shell-Syntaxcheck:
```bash
bash -n scripts/slurm/*.sh
```

2. Pfad-Check:
- Die SLURM-Skripte `cd` ins Repo-Root (`.../arlab_cv_training`)
- Die `srun python3 ...` Aufrufe zeigen auf `scripts/local/*.py`

## 3) SLURM Submit (nur wenn Voraussetzungen erfüllt)
Beispiele:
```bash
sbatch scripts/slurm/slurm_train_yolo11.sh
sbatch scripts/slurm/slurm_compare_models.sh
```

Voraussetzungen:
- `yolo_weights/yolo11n-seg.pt` bzw. `yolo_weights/yolo26n-seg.pt` vorhanden
- Dataset-Ordner inkl. `data.yaml` vorhanden (z.B. `data_640/data.yaml`)

