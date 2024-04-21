import numpy as np
from PIL import Image
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from torch_geometric.datasets import MoleculeNet

def plot_molecule(num_mols):
    
    data = MoleculeNet(root='GNN/data', name='ESOL')
    
    for i in tqdm(range(num_mols)):
        idx = np.random.randint(0, len(data))
        mol = data[idx]
        smiles = mol.smiles
        plottable_mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(plottable_mol, f"misc/mol_plots/mol_{i}.png",
                    size=(400, 400), firImage=True)
        
        # Open the image file and convert it to RGB
        img = Image.open(f"misc/mol_plots/mol_{i}.png").convert("RGB")
        
        if len(smiles) > 35:
            smiles = smiles[:35] + "..."
            
        plt.figure(figsize=(6, 6))
        plt.title(f"Molecule: {smiles}\nlog(Solubility): {mol.y.item()} mol/L", y=0.8)
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(f"misc/mol_plots/mol_{i}.png")
        plt.close('all')
        
    return 0

if __name__ == '__main__':
    plot_molecule(5)