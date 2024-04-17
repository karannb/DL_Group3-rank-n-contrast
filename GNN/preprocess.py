import random
import pickle
import pandas as pd
from deepchem.molnet import load_delaney
from torch_geometric.datasets import MoleculeNet
from torch_geometric.utils import from_smiles

def preprocess(lib='pyg'):
    '''
    Create a csv file with the following columns:
    path: path to the pickled molecule
    solubility: solubility of the molecule (normalized to mean 0, std 1)
    split: train/valid/test
    '''

    if lib == "deepchem":
        '''
        DOES NOT WORK!!
        '''
        _, data, _ = load_delaney(splitter='random',
                                reload=False)

        train, valid, test = data
        labels = []
        paths = []
        split = []
        count = 0

        for mol in train.ids:
            cur_mol = from_smiles(mol)
            with open(f'GNN/data/ESOL/mol_{count}', 'wb') as f:
                pickle.dump(cur_mol, f)
            paths.append(f'./data/ESOL/mol_{count}')
            labels.append(train.y[count][0])
            split.append("train")
            count += 1

        for mol in valid.ids:
            cur_mol = from_smiles(mol)
            with open(f'GNN/data/ESOL/mol_{count}', 'wb') as f:
                pickle.dump(cur_mol, f)
            paths.append(f'GNN/data/ESOL/mol_{count}')
            labels.append(valid.y[count-len(train.ids)][0])
            split.append("valid")
            count += 1

        for mol in test.ids:
            cur_mol = from_smiles(mol)
            with open(f'GNN/data/ESOL/mol_{count}', 'wb') as f:
                pickle.dump(cur_mol, f)
            paths.append(f'GNN/data/ESOL/mol_{count}')
            labels.append(test.y[count-len(train.ids)-len(valid.ids)][0])
            split.append("test")
            count += 1

        # Save the df
        df = pd.DataFrame({'path': paths, 'solubility': labels, 'split': split})
        df.to_csv('GNN/data/esol.csv', index=False)

    elif lib == "pyg":

        data = MoleculeNet("data/", "ESOL")
        split_inds = list(range(len(data)))
        random.shuffle(split_inds)
        train_inds = split_inds[:int(0.8*len(data))]
        val_inds = split_inds[int(0.8*len(data)):int(0.9*len(data))]
        test_inds = split_inds[int(0.9*len(data)):]

        # Find mean and std of train set mols to normalize labels
        mean = 0
        std = 0
        for i in train_inds:
            mol = data[i]
            mean += mol.y[0].item()
        mean /= len(train_inds)
        for i in train_inds:
            mol = data[i]
            std += (mol.y[0].item() - mean)**2
        std = (std/len(train_inds))**0.5

        paths = []
        labels = []
        split = []
        count = 0

        for i in train_inds:
            mol = data[i]
            with open(f'data/ESOL/mol_{count}', 'wb') as f:
                pickle.dump(mol, f)
            paths.append(f'data/ESOL/mol_{count}')
            labels.append((mol.y[0].item() - mean)/std)
            split.append("train")
            count += 1
        
        for i in val_inds:
            mol = data[i]
            with open(f'data/ESOL/mol_{count}', 'wb') as f:
                pickle.dump(mol, f)
            paths.append(f'data/ESOL/mol_{count}')
            labels.append((mol.y[0].item() - mean)/std)
            split.append("valid")
            count += 1

        for i in test_inds:
            mol = data[i]
            with open(f'data/ESOL/mol_{count}', 'wb') as f:
                pickle.dump(mol, f)
            paths.append(f'data/ESOL/mol_{count}')
            labels.append((mol.y[0].item() - mean)/std)
            split.append("test")
            count += 1

        # Save the df
        df = pd.DataFrame({'path': paths, 'solubility': labels, 'split': split})
        df.to_csv('data/esol.csv', index=False)

    return

if __name__ == '__main__':
    preprocess()