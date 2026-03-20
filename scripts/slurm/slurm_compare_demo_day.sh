#!/usr/bin/env bash

# ============================================================
# Model comparison for the `data_640_demo_day` dataset
# ============================================================

#SBATCH --job-name=compare-demo-day
#SBATCH --partition=epyc-gpu
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aleksander1.michalak@uni-a.de
#SBATCH --time=1:00:00
#SBATCH --output=/hpc/gpfs2/home/u/michalek/arlab_cv_training/logs/comparison/compare_demo_day_%j.out
#SBATCH --error=/hpc/gpfs2/home/u/michalek/arlab_cv_training/logs/comparison/compare_demo_day_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /hpc/gpfs2/home/u/michalek/arlab_cv_training
mkdir -p logs/comparison
source .venv/bin/activate

echo "=========================================="
echo "📊 Model Comparison - Demo Day Dataset"
echo "=========================================="

srun python3 scripts/local/compare_models_demo_day.py
