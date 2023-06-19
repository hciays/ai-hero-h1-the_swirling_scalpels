#!/bin/bash

#SBATCH --job-name=AI-HERO_health_baseline_evaluation
#SBATCH --partition=cpuonly
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=152
#SBATCH --time=20:00:00
#SBATCH --output=/home/hk-project-test-aihero2/hgf_pdv3669/baseline_evaluation.txt

export CUDA_CACHE_DISABLE=1
export OMP_NUM_THREADS=1

group_workspace=/home/hk-project-test-aihero2/hgf_pdv3669

source ${group_workspace}/health_baseline_env/bin/activate
python ${group_workspace}/ai-hero2/eval.py --pred_dir ./pred