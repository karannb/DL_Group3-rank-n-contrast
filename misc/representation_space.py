import torch
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader

import umap
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from GNN.dataset import ESOL
from GNN.model import GCNEncoder, GCNMLP, ModelArgs

def plot_representation_space(normal_model,
                              rnc_model,
                              data_loader,
                              split='train'):
    
    # Load models
    model_args = ModelArgs()
    model_args.in_channels = 9
    model_args.out_channels = 100
    model_args.hidden_channels = 64
    model_args.num_layers = 5
    model_args.dropout = 0.0
    
    normal_state_dict = torch.load(normal_model, map_location='cpu')['model']
    rnc_state_dict = torch.load(rnc_model, map_location='cpu')['model']
    
    # Assign empty model
    normal_model = GCNMLP(model_args)
    rnc_model = GCNEncoder(model_args)
    
    # Load state dict
    normal_model.load_state_dict(normal_state_dict)
    rnc_model.load_state_dict(rnc_state_dict)
    
    # Only use the encoder
    normal_model = normal_model.encoder
    
    # Set models to eval mode
    normal_model.eval()
    rnc_model.eval()
    
    normal_representation_space = []
    rnc_representation_space = []
    with torch.no_grad():
        for (mol, _) in tqdm(data_loader):
            normal_representation_space.append(normal_model(mol).numpy())
            rnc_representation_space.append(rnc_model(mol).numpy())
    
    normal_representation_space = np.concatenate(normal_representation_space, axis=0)
    rnc_representation_space = np.concatenate(rnc_representation_space, axis=0)
    
    # train a t-SNE model on both representation spaces
    normal_representation_space = umap.UMAP().fit_transform(normal_representation_space)
    rnc_representation_space = umap.UMAP().fit_transform(rnc_representation_space)
    
    # Plot subplots and save
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].scatter(normal_representation_space[:, 0], normal_representation_space[:, 1])
    axes[0].set_title('Normal GNN')
    axes[1].scatter(rnc_representation_space[:, 0], rnc_representation_space[:, 1])
    axes[1].set_title('RnC GNN')
    plt.savefig(f'misc/mol_plots/representation_space_{split}.png')
    plt.close('all')
    
    return 0

if __name__ == '__main__':
    rnc_model = "GNN/save/ESOL_models/RnC_GNN_ESOL_ep_800_lr_0.01_d_0.0_wd_1e-05_mmt_0.9_bsz_256_temp_2_label_l1_feature_l2_trial_5/ckpt_epoch_800.pth"
    normal_model = "GNN/save/ESOL_models/L1_ESOL_GNN_ep_800_lr_0.01_d_0.0_wd_1e-06_mmt_0.9_bsz_128_aug_crop,flip,color,grayscale_trial_6/ckpt_epoch_800.pth"
    split = 'train'
    esol_data = ESOL("GNN/data", split, use2plot=True)
    dataloader = DataLoader(esol_data, batch_size=1, shuffle=False)
    
    plot_representation_space(normal_model, rnc_model, dataloader, split)