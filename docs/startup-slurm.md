# Training Pipeline (SLURM)

Most students and day-to-day use should follow **`docs/startup-local.md`** (local Python scripts). This page is for **optional** use on **batch clusters** with SLURM—for example the **LICCA** cluster at the University of Augsburg—where you submit GPU jobs instead of running training interactively on a laptop.

## 1) General pattern
The repo contains ready-to-submit SLURM scripts (`scripts/slurm/slurm_train_*.sh`, `scripts/slurm/slurm_compare_*.sh`).
They typically:
1. `cd` into the repo folder
2. activate `.venv`
3. run the matching Python script via `srun python3 ...`

## 2) Train jobs
Examples:
- YOLO11 training:
  ```bash
  sbatch scripts/slurm/slurm_train_yolo11.sh
  ```
- YOLO26 training:
  ```bash
  sbatch scripts/slurm/slurm_train_yolo26.sh
  ```
- Demo-day training:
  ```bash
  sbatch scripts/slurm/slurm_train_demo_day.sh
  ```

## 3) Compare jobs
- Full comparison:
  ```bash
  sbatch scripts/slurm/slurm_compare_models.sh
  ```
- Demo-day comparison:
  ```bash
  sbatch scripts/slurm/slurm_compare_demo_day.sh
  ```

## 4) Notes / prerequisites
- Datasets: copy or clone the repo on the cluster, then extract **`datasets.zip`** from the repo root (same layout as local: dataset folders with `data.yaml` next to `scripts/`). See `docs/startup-local.md` section “Data (datasets)”.
- SLURM jobs assume those dataset directories already exist on disk at the repo path used in the job.
- `scripts/local/train_models.py` requires pretrained weights in `yolo_weights/`:
  - `yolo_weights/yolo11n-seg.pt` for `--model yolo11n-seg`
  - `yolo_weights/yolo26n-seg.pt` for `--model yolo26n-seg`
- Output logs are written into `logs/` (and SLURM may additionally create `slurm-*.out` files depending on cluster config).

