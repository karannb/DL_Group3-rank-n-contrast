import os
from rdkit import Chem
import pandas as pd

import torch
from torch.utils.data import Dataset

class QM7(Dataset):

    def __init__(self, root):

        super(QM7, self).__init__()
        label_path = os.path.join(root, 'gdb7.sdf.csv')
        self.labels = pd.read_csv(label_path, index_col=None)
        mol_path = os.path.join(root, 'gdb7.sdf')
        self.mols = Chem.SupplierFromFilename(mol_path)

        print('Number of molecules:', len(self.mols))

    def __len__(self):

        return len(self.mols)
    
    def __getitem__(self, idx):

        mol = self.mols[idx]
        label = self.labels.iloc[idx]

        return mol, torch.tensor(label, dtype=torch.float32)

if __name__ == '__main__':

    qm7 = QM7('data/gdb7')
    mol0, lbl0 = qm7[0]
    print([x for x in dir(mol0) if '__' not in x])
    for atom in mol0.GetAtoms():
        print(atom.GetSymbol(), atom.GetIdx())