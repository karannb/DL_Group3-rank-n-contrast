#!/bin/sh

#SBATCH --job-name=GNN
#SBATCH --output=logs/l1_GNN.out
#SBATCH --error=logs/l1_GNN.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

export CUDA_VISIBLE_DEVICES=2

cd GNN/

python3 main_l1.py --data_folder data/

# other possible runs are
# python3 main_l1.py --data_folder data/, with output=logs/L1_GNN.out and error=logs/L1_GNN.err
# python3 main_rnc.py --data_folder data/, with output=logs/RnC_GNN.out and error=logs/RnC_GNN.err

# python3 main_linear.py --data_folder data/ --loss [L1(default)/L2/huber] --ckpt <path> --freeze_encoder, 
# with output=logs/linear_GNN.out and error=logs/linear_GNN.err