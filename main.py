from loss import RnCLoss
from dataset import ESOL
from models.GNN import GCNEncoder, GCNMLP

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:

    def __init__(self, root, batch_size=100, lr=1e-4, weight_decay=1e-6, temperature=0.07, label_diff='l1', feature_sim='l2'):
        
        self.dataset = ESOL(root, clip=True)
        self.train_loader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        self.criteria = torch.nn.L1Loss() #temperature=temperature, label_diff=label_diff, feature_sim=feature_sim

        self.encoder = GCNMLP()

    def train(self, epochs=1000):

        self.encoder.train()
        self.encoder.to(device)

        optimizer = optim.Adam(self.encoder.parameters(), lr=1e-4, weight_decay=1e-6)

        for epoch in range(epochs):
            for i, (mols, labels) in enumerate(self.train_loader):

                optimizer.zero_grad()

                features = self.encoder(mols)
                loss = self.criteria(features, labels)

                loss.backward()
                optimizer.step()

                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, i * len(mols), len(self.train_loader.dataset),
                    100. * i / len(self.train_loader), loss.item()))

        print('Training done!')
        return
    
if __name__ == '__main__':
    trainer = Trainer('data')
    trainer.train()