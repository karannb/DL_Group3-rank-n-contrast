#!/bin/sh

#SBATCH --job-name=run
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G

cd GNN/

python3 main_l1.py