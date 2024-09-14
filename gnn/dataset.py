from torch.utils.data import Dataset
from rdkit import Chem
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_splitted_lipo_dataset(ratios=[0.7, 0.3, 0], seed=123):
    data_path = '../data/'

    chembl_data = pd.read_csv(data_path + 'train.csv') # Open original dataset
    train_data = chembl_data[['Smiles', 'IC50_nM', 'pIC50']]
    test_data = pd.read_csv(data_path + 'test.csv')

    # train_val, test = train_test_split(raw_data, test_size=ratios[2], random_state=seed)
    train, val = train_test_split(train_data, test_size=ratios[1]/(ratios[0]+ratios[1]), random_state=seed)
    
    return train, val, test_data

LIST_SYMBOLS = ['C', 'N', 'O', 'S', 'F', 'H', 'Si', 'P', 'Cl', 'Br',
            'Li', 'Na', 'K', 'Mg', 'Ca', 'Fe', 'As', 'Al', 'I', 'B',
            'V', 'Tl', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn',
            'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'Mn', 'Cr', 'Pt', 'Hg', 'Pb']


def atom_feature(atom):
    return np.array(char_to_ix(atom.GetSymbol(), LIST_SYMBOLS) +
                    char_to_ix(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
                    char_to_ix(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    char_to_ix(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    char_to_ix(int(atom.GetIsAromatic()), [0, 1]))    # (40, 6, 5, 6, 2)


def char_to_ix(x, allowable_set):
    if x not in allowable_set:
        return [0] # Unknown Atom Token
    return [allowable_set.index(x)+1]


def mol2graph(smi, MAX_LEN):
    mol = Chem.MolFromSmiles(smi)

    X = np.zeros((MAX_LEN, 5), dtype=np.uint8)
    A = np.zeros((MAX_LEN, MAX_LEN), dtype=np.uint8)

    temp_A = Chem.rdmolops.GetAdjacencyMatrix(mol).astype(np.uint8, copy=False)[:MAX_LEN, :MAX_LEN]
    num_atom = temp_A.shape[0]
    A[:num_atom, :num_atom] = temp_A + np.eye(temp_A.shape[0], dtype=np.uint8)
    
    for i, atom in enumerate(mol.GetAtoms()):
        feature = atom_feature(atom)
        X[i, :] = feature
        if i + 1 >= num_atom: break
            
    return X, A

class gcnDataset(Dataset):
    def __init__(self, df, max_len=120):
        self.smiles = df["Smiles"]
        self.exp = df["pIC50"].values
                
        list_X = list()
        list_A = list()
        for i, smiles in enumerate(self.smiles):
            X, A = mol2graph(smiles, max_len)
            list_X.append(X)
            list_A.append(A)
            
        self.X = np.array(list_X, dtype=np.uint8)
        self.A = np.array(list_A, dtype=np.uint8)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.A[index], self.exp[index]