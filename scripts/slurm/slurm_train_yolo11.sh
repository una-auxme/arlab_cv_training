#!/usr/bin/env bash

# Use a job name that describes your job (not too long)
#SBATCH --job-name=yolo11n-seg-train

# Select a partiton (epyc, epyc-mem or epyc-gpu)
#SBATCH --partition=epyc-gpu

# Request memory (default 512M)
#SBATCH --mem=32G

# CPU cores per task (für DataLoader num_workers)
#SBATCH --cpus-per-task=16

# Request GPU (1×Nvidia A100)
#SBATCH --gres=gpu:a100:1

# Events when a mail is sent
#SBATCH --mail-type=FAIL

# Send mail to this address. Fill in valid mail address or delete this line.
#SBATCH --mail-user=aleksander1.michalak@uni-a.de

# Timelimit 2 days (für YOLO Training mit 4 Experimenten)
#SBATCH --time=2-00:00:00

# Set number of OpenMP threads (1 pro Worker, um Over-subscription zu vermeiden)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Output und Error Logs:
#SBATCH --output=/hpc/gpfs2/home/u/michalek/arlab_cv_training/logs/training/train_yolo11_%j.out
#SBATCH --error=/hpc/gpfs2/home/u/michalek/arlab_cv_training/logs/training/train_yolo11_%j.err

# Wechsle ins Arbeitsverzeichnis (wichtig für relative Pfade im Skript)
cd /hpc/gpfs2/home/u/michalek/arlab_cv_training

# Erstelle logs-Verzeichnis falls nicht vorhanden
mkdir -p logs/training

# Aktiviere venv
source .venv/bin/activate

# Führe das YOLO Training aus mit yolo11n-seg Modell
# Das Skript trainiert 4 Experimente nacheinander (no_augmentation, baseline, moderate_geom, strong_geom)
# --dataset: data_420, data_640 oder fruit_dataset_640 (default: data_640)
srun python3 scripts/local/train_models.py --model yolo11n-seg --dataset data_640
