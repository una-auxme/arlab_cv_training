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

# Change to the repo working directory (important for relative paths in the Python script)
cd /hpc/gpfs2/home/u/michalek/arlab_cv_training

# Create logs directory if it doesn't exist
mkdir -p logs/comparison

# Activate virtual environment
source .venv/bin/activate

# Run the model comparison script.
# It evaluates all trained models (yolo11n-seg and yolo26n-seg) across the 4 augmentation variants
# and writes a comparison table.
srun python3 scripts/local/compare_models.py
