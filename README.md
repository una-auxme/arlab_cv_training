# arlab_cv_training

Trainieren und Vergleichen von YOLO-Segmentation-Varianten (Ultralytics) für unterschiedliche Datensätze und Augmentations-Settings. Dieses Projekt wurde speziell fuer das `ARLab` der Universitat Augsburg gebaut und im Rahmen der Entwicklung des `Zirbi`-Roboters verwendet.

## Quickstart
- Lokales Setup/Training: `docs/startup-local.md`
- SLURM Jobs: `docs/startup-slurm.md`

Python Entry Points:
- `scripts/local/train_models.py`

SLURM Job Scripts:
- `scripts/slurm/slurm_train_*.sh`, `scripts/slurm/slurm_compare_*.sh`
