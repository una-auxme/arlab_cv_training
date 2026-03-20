#!/usr/bin/env bash

# Use a job name that describes your job (not too long)
#SBATCH --job-name=compare-models

# Select a partiton (epyc, epyc-mem or epyc-gpu)
#SBATCH --partition=epyc-gpu

# Request memory (default 512M)
# Evaluation benötigt weniger Memory als Training
#SBATCH --mem=16G

# CPU cores per task (für DataLoader num_workers)
#SBATCH --cpus-per-task=8

# Request GPU (1×Nvidia A100)
#SBATCH --gres=gpu:a100:1

# Events when a mail is sent
#SBATCH --mail-type=FAIL,END

# Send mail to this address. Fill in valid mail address or delete this line.
#SBATCH --mail-user=aleksander1.michalak@uni-a.de

# Timelimit (Evaluation ist schneller als Training, 1 Stunde sollte reichen)
#SBATCH --time=1:00:00

# Set number of OpenMP threads (1 pro Worker, um Over-subscription zu vermeiden)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Output und Error Logs:
#SBATCH --output=/hpc/gpfs2/home/u/michalek/arlab_cv_training/logs/comparison/compare_models_%j.out
#SBATCH --error=/hpc/gpfs2/home/u/michalek/arlab_cv_training/logs/comparison/compare_models_%j.err

# Wechsle ins Arbeitsverzeichnis (wichtig für relative Pfade im Skript)
cd /hpc/gpfs2/home/u/michalek/arlab_cv_training

# Erstelle logs-Verzeichnis falls nicht vorhanden
mkdir -p logs/comparison

# Aktiviere venv
source .venv/bin/activate

# Führe das Vergleichsskript aus
# Das Skript evaluiert alle trainierten Modelle (yolo11n-seg und yolo26n-seg)
# mit allen 4 Augmentation-Varianten und erstellt eine Vergleichstabelle
srun python3 scripts/local/compare_models.py
