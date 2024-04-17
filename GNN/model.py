'''
Simple Convolutional GNN model.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool

class ModelArgs:
    in_channels = 9
    out_channels = 128
    hidden_channels = 64
    num_layers = 4
    dropout = 0.5

class Encoder(nn.Module):

    def __init__(self, args : ModelArgs):
        super(Encoder, self).__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.num_layers = args.num_layers

        self.convs.append(GINConv(nn.Sequential(nn.Linear(args.in_channels, 4*args.hidden_channels), 
                                                nn.ReLU(), 
                                                nn.Linear(4*args.hidden_channels, args.hidden_channels))))
        self.bns.append(nn.BatchNorm1d(args.hidden_channels))
        for _ in range(self.num_layers-2):
            self.convs.append(GINConv(nn.Sequential(nn.Linear(args.hidden_channels, 4*args.hidden_channels), 
                                                    nn.ReLU(), 
                                                    nn.Linear(4*args.hidden_channels, args.hidden_channels))))
            self.bns.append(nn.BatchNorm1d(args.hidden_channels))
        
        self.convs.append(GINConv(nn.Sequential(nn.Linear(args.hidden_channels, args.out_channels), 
                                                nn.ReLU(), 
                                                nn.Linear(args.out_channels, args.out_channels))))
        
        self.dropout = args.dropout

        # Get number of parameters
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print(f'Number of parameters: {num_params}')

    def forward(self, mol):

        x, edge_index = mol.x, mol.edge_index
        x = x.float()

        for i in range(self.num_layers-1):
            x = self.convs[i](x=x, edge_index=edge_index)
            x = self.bns[i](x)
            x = F.tanh(x) # works better than ReLU
            # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x=x, edge_index=edge_index)

        x = torch.cat([global_add_pool(x, mol.batch), global_mean_pool(x, mol.batch)], dim=1)
        return x
    
class GCNEncoder(nn.Module):
        
    def __init__(self, args : ModelArgs):
        super(GCNEncoder, self).__init__()
        self.encoder = Encoder(args)

    def forward(self, mol):
        return self.encoder(mol)

class GCNMLP(GCNEncoder):
    
    def __init__(self, args : ModelArgs):
        super(GCNMLP, self).__init__(args)
        self.mlp = nn.Sequential(nn.Linear(2*args.out_channels, args.out_channels),
                                nn.ReLU(),
                                nn.Linear(args.out_channels, 512),
                                nn.ReLU(),
                                nn.Linear(512, 1))

    def forward(self, mol):
        x = super(GCNMLP, self).forward(mol)
        x = self.mlp(x)
        return x