# Training Pipeline (SLURM)

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
- SLURM scripts assume the dataset directories and `data.yaml` exist on disk.
- `scripts/local/train_models.py` requires pretrained weights in `yolo_weights/`:
  - `yolo_weights/yolo11n-seg.pt` for `--model yolo11n-seg`
  - `yolo_weights/yolo26n-seg.pt` for `--model yolo26n-seg`
- Output logs are written into `logs/` (and SLURM may additionally create `slurm-*.out` files depending on cluster config).

