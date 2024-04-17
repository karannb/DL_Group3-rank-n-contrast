#!/bin/sh

#SBATCH --job-name=run
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

export CUDA_VISIBLE_DEVICES=1

cd GNN/

python3 CNN/main_l1.py --data_folder ./CNN/data