'''
Simple Convolutional GNN model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCNEncoder(nn.Module):

    def __init__(self, in_channels=9, out_channels=128, hidden_channels=64, num_layers=1, dropout=0.5):
        super(GCNEncoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = num_layers

        self.convs.append(GCNConv(in_channels, hidden_channels, normalize=False))
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

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
    
        def __init__(self, in_channels=9, out_channels=128, hidden_channels=64, num_layers=3, dropout=0.5):
            super(GCNMLP, self).__init__(in_channels, out_channels, hidden_channels, num_layers, dropout)
            self.mlp = nn.Sequential(
                nn.Linear(out_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, 1)
            )
    
        def forward(self, mol):
            x = super(GCNMLP, self).forward(mol)
            x = global_mean_pool(x, mol.batch)
            x = self.mlp(x)
            return x