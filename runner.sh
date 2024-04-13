#!/bin/sh

#SBATCH --job-name=run
#SBATCH --output=run.out
#SBATCH --error=run.err
#SBATCH --time=00:10:00
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

cd CNN/

python3 main_l1.py