# arlab_cv_training

Train and compare YOLO segmentation variants (Ultralytics) for different datasets and augmentation settings. This project was built specifically for the `ARLab` at the University of Augsburg and developed in the context of the `Zirbi` robot.

## Datasets
Training data is shipped as a single archive in the repository root: **`datasets.zip`**. After cloning, extract it here (repo root) so the dataset folders (e.g. `data_640/`, each with a `data.yaml`) sit alongside `scripts/` and `docs/`. Details: `docs/startup-local.md`.

## Quickstart
The intended workflow for most users is **local training** via `scripts/local/` (see `docs/startup-local.md`). **SLURM** scripts under `scripts/slurm/` are optional: they are provided for running jobs on university **compute clusters** (e.g. **LICCA** at the University of Augsburg) and are documented in `docs/startup-slurm.md`.

- Local setup / training (primary): `docs/startup-local.md`
- SLURM / cluster jobs (optional): `docs/startup-slurm.md`

Python entry points:
- `scripts/local/train_models.py`

SLURM job scripts (cluster only):
- `scripts/slurm/slurm_train_*.sh`, `scripts/slurm/slurm_compare_*.sh`
