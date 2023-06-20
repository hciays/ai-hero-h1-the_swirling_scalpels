#!/bin/bash

#SBATCH --job-name=AI-HERO_health_inference_h1
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=152
#SBATCH --time=20:00:00
#SBATCH --output=/hkfs/work/workspace/scratch/hgf_pdv3669-H1/inference_h1.txt

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=1

group_workspace=/hkfs/work/workspace/scratch/hgf_pdv3669-H1

source ${group_workspace}/health_h1/bin/activate
python ${group_workspace}/ai-hero-h1-the_swirling_scalpels/inference.py --from_checkpoint ./lightning_logs/version_3/checkpoints/epoch=99-step=10000.ckpt --pred_dir ./pred