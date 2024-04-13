'''
Simple Convolutional GNN model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class ModelArgs:
    in_channels = 9
    out_channels = 128
    hidden_channels = 64
    num_layers = 3
    dropout = 0.5

class GCNEncoder(nn.Module):

    def __init__(self, args : ModelArgs):
        super(GCNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = args.num_layers

        self.convs.append(GCNConv(args.in_channels, args.hidden_channels, normalize=False))
        self.bns.append(nn.BatchNorm1d(args.hidden_channels))
        for _ in range(args.num_layers - 2):
            self.convs.append(GCNConv(args.hidden_channels, args.hidden_channels, normalize=False))
            self.bns.append(nn.BatchNorm1d(args.hidden_channels))
        self.convs.append(GCNConv(args.hidden_channels, args.out_channels, normalize=False))

        self.dropout = args.dropout

    def forward(self, mol):

        x, edge_index, edge_attr = mol.x, mol.edge_index, mol.edge_attr
        x = x.float()

        for i in range(self.num_layers - 1):
            x = self.convs[i](x=x, edge_index=edge_index, edge_weight=torch.mean(edge_attr, dim=-1, dtype=torch.float32))
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x
    
class GCNMLP(GCNEncoder):
    
        def __init__(self, args : ModelArgs):
            super(GCNMLP, self).__init__(args)
            self.mlp = nn.Sequential(
                nn.Linear(args.out_channels, args.hidden_channels),
                nn.ReLU(),
                nn.Linear(args.hidden_channels, 1)
            )
    
        def forward(self, mol):
            x = super(GCNMLP, self).forward(mol)
            x = global_mean_pool(x, mol.batch)
            x = self.mlp(x)
            return x