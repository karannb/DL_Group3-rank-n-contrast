import pickle
import pandas as pd
from deepchem.molnet import load_delaney
from torch_geometric.utils import from_smiles

def preprocess():

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
        paths.append(f'GNN/data/ESOL/mol_{count}')
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

    return

if __name__ == '__main__':
    preprocess()