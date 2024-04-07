import os
import pickle
from deepchem.molnet import load_delaney
from torch_geometric.utils import from_smiles

import torch
from torch.utils.data import Dataset

class ESOL(Dataset):

    def __init__(self, root, clip=False):

        super(ESOL, self).__init__()
        _, self.data, _ = load_delaney(splitter='random',
                                       data_dir=root,
                                       save_dir=root,
                                       reload=False)
        
        # Convert from smiles to molecule
        train, valid, test = self.data
        train_mols = []
        for mol in train.ids:
            train_mols.append(from_smiles(mol))
        valid_mols = []
        for mol in valid.ids:
            valid_mols.append(from_smiles(mol))
        test_mols = []
        for mol in test.ids:
            test_mols.append(from_smiles(mol))

        self.data_dict = {
            'train': (train_mols, train.y),
            'valid': (valid_mols, valid.y),
            'test': (test_mols, test.y)
        }

        self.mode = 'train'
        print('''Loaded ESOL dataset:
              - train size: {}
              - valid size: {}
              - test size: {}'''.format(len(self.data_dict['train'][0]), len(self.data_dict['valid'][0]), len(self.data_dict['test'][0])))

    def __len__(self):

       assert self.mode == 'train' or self.mode == 'valid' or self.mode == 'test', 'Invalid mode, possible values are train, valid, test'

       return len(self.data_dict[self.mode][0])
    
    def __getitem__(self, idx):

        assert self.mode == 'train' or self.mode == 'valid' or self.mode == 'test', 'Invalid mode, possible values are train, valid, test'
        
        mol = self.data_dict[self.mode][0][idx]
        label = self.data_dict[self.mode][1][idx]
        return mol, torch.tensor(label, dtype=torch.float32)
    
    def train(self):
        self.mode = 'train'

    def eval(self):
        self.mode = 'valid'

    def test(self):
        self.mode = 'test'

if __name__ == '__main__':

    esol = ESOL('data/')
    mol0, lbl0 = esol[1]
    print(mol0)