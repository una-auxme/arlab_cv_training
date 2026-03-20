#!/usr/bin/env bash

# ============================================================
# YOLO11n-seg Training auf data_640_demo_day mit strong_geom_fruit
# ============================================================

#SBATCH --job-name=yolo11-demo-day
#SBATCH --partition=epyc-gpu
#SBATCH --mem=32G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=aleksander1.michalak@uni-a.de
#SBATCH --time=2-00:00:00
#SBATCH --output=/hpc/gpfs2/home/u/michalek/arlab_cv_training/logs/training/train_demo_day_%j.out
#SBATCH --error=/hpc/gpfs2/home/u/michalek/arlab_cv_training/logs/training/train_demo_day_%j.err

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd /hpc/gpfs2/home/u/michalek/arlab_cv_training
mkdir -p logs/training
source .venv/bin/activate

echo "=========================================="
echo "🚀 YOLO11n-seg Training"
echo "📊 Dataset:      data_640_demo_day"
echo "🔧 Augmentation: strong_geom_fruit"
echo "=========================================="

srun python3 scripts/local/train_models.py --model yolo11n-seg --dataset data_640_demo_day --augmentation strong_geom_fruit
