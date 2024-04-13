import os
import pickle
import pandas as pd

import torch
from torch.utils.data import Dataset

class ESOL(Dataset):

    def __init__(self, data_folder, split, clip=False):

        super(ESOL, self).__init__()
        df = pd.read_csv(f'./GNN/data/esol.csv')
        self.df = df[df['split'] == split]
        self.split = split
        self.data_folder = data_folder

        print('''Loaded ESOL dataset:
              - data size: {}'''.format(len(self.df)))

    def __len__(self):

       return len(self.df)
    
    def __getitem__(self, idx):

        with open(self.df.iloc[idx]["path"], "rb") as f:
            mol = pickle.load(f)
        label = self.df.iloc[idx]["solubility"]
        return mol, torch.tensor(label, dtype=torch.float32)

# if __name__ == '__main__':

#     esol = ESOL('data/')
#     mol0, lbl0 = esol[1]
#     print(mol0)