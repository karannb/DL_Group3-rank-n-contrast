#!/bin/sh

#SBATCH --job-name=GNN
#SBATCH --output=logs/linear_GNN.out
#SBATCH --error=logs/linear_GNN.err
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=128G

export CUDA_VISIBLE_DEVICES=2

cd GNN/

python3 main_linear.py --data_folder data/ \
 --ckpt save/ESOL_models/RnC_GNN_ESOL_ep_400_lr_4e-05_d_0_wd_0.0001_mmt_0.9_bsz_256_temp_2_label_l1_feature_l2_trial_0/ckpt_epoch_400.pth \
 --freeze_encoder

# other possible runs are
# python3 main_l1.py --data_folder data/, with output=logs/L1_GNN.out and error=logs/L1_GNN.err
# python3 main_rnc.py --data_folder data/, with output=logs/RnC_GNN.out and error=logs/RnC_GNN.err

# python3 main_linear.py --data_folder data/ --loss [L1(default)/L2/huber] --ckpt <path>, 
# with output=logs/linear_GNN.out and error=logs/linear_GNN.err