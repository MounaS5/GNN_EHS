import rdkit
from rdkit import Chem

def listOfAtoms(df):
  smilesList = df['SMILES'].to_list()
  molList =[Chem.MolFromSmiles(i) for i in smilesList]
  atomsList = [[atom.GetSymbol()for atom in m.GetAtoms()] for m in molList] 
  flat_ls = [item for sublist in atomsList for item in sublist]
  atom_types = list(set(flat_ls))
  return atom_types, Counter(flat_ls)

#possible_atom_list = ['C', 'O', 'Cl', 'N', 'F', 'Br', 'S','P' ,'Ti','Ga','As','Se','Mo','Ni','In','Hg','V','Pb','Te','Unknown']
possible_atom_degree = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # (getDegree()
print(possible_atom_degree )
possible_numH_list = [0, 1, 2, 3, 4]
possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
possible_valency = [0, 1, 2, 3, 4, 5, 6]

possible_hybridization = [Chem.rdchem.HybridizationType.SP, 
                          Chem.rdchem.HybridizationType.SP2,
                          Chem.rdchem.HybridizationType.SP3, 
                          Chem.rdchem.HybridizationType.SP3D, 
                          Chem.rdchem.HybridizationType.SP3D2]

possible_num_bonds = [0,1,2,3,4,5,6]  #atom_valence

possible_number_radical_e_list = [0, 1, 2]
#possible_chirality_list = ['R', 'S']

## Why do you need them if you already have a one hot encoding?


#########################
# --- Atom features --- #
#########################

## Node related entities of a graph
def one_of_k_encoding(x, allowable_set):
  if x not in allowable_set:
    raise Exception("input {0} not in allowable set{1}:".format(
        x, allowable_set))
  return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x,allowable_set):
    if x not in allowable_set:
        x=allowable_set[-1]

    return list(map(lambda s: x==s, allowable_set))

Total_atom_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 
                   'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']

def atom_features(atom,
                  bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False): #takes atom object as an input

# Structure : (atom type + degree + formal charge + radical electrons+ hybridisation + aromaticity)
# Structure + (is it required to mention H)+ (is it necessary to use chirality)

    Symbol =atom.GetSymbol()

    Type_atom = one_of_k_encoding_unk(Symbol,Total_atom_list)
    Degree = one_of_k_encoding(atom.GetDegree(),possible_atom_degree)                           
    Valency = one_of_k_encoding(atom.GetImplicitValence(), possible_valency)
    FormalCharge_RadicalNum = [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
    Aromaticity = [atom.GetIsAromatic()]
    Hybridization = one_of_k_encoding_unk(atom.GetHybridization(), possible_hybridization)
    #Bonds_atom    = one_of_k_encoding_unk(len(atom.GetNeighbors()), possible_num_bonds)
    if bool_id_feat:
      return np.array([atom_to_id(atom)])
    else:
      results = Type_atom + Degree + Valency + \
      FormalCharge_RadicalNum + Aromaticity + Hybridization 
        #Bonds_atom ++> better to give formal charge

    if not explicit_H:
      results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                possible_numH_list)
    if use_chirality:
      try:
        results = results + one_of_k_encoding_unk(
            atom.GetProp('_CIPCode'),
            ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
      except:
        results = results + [False, False
                            ] + [atom.HasProp('_ChiralityPossible')]

    
#Ring_atom +
# The list 'results' is returned as an numpy array object
    return np.array(results).astype(np.float32)

              

#########################
# --- Bond features --- #
#########################

## Edge related entities of a graph
def get_bond_pair(mol):
    bonds = mol.GetBonds()#Returns a read-only sequence of the atomÃ¢â‚¬â„¢s bonds
    res = [[],[]]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(),bond.GetBeginAtomIdx()]
    return res

def bond_features(bond,use_chirality=False):
    bt = bond.GetBondType()

    # bond features is an array of 0s and 1s. 
    #If a particular feature exists it gives , otherwise it gives zero.

    bond_feats = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()]
    if use_chirality:
      bond_feats = bond_feats + one_of_k_encoding_unk(
      str(bond.GetStereo()),
        ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return np.array(bond_feats).astype(np.float32)

###################################
# --- Molecule to torch graph --- #
###################################

def mol2torchdata(mol):

    atoms = mol.GetAtoms()#all atoms in a molecule
    bonds = mol.GetBonds()#all bonds in a molecule


    # Information on nodes
    node_f = [atom_features(atom) for atom in atoms]
# Invoke atom_features function-assign array uniques to each atom


    # Information on edges
    edge_index = get_bond_pair(mol)#pairs of bonds in a molecule
    edge_attr  = []
    #return an array of bond features
    #print(edge_attr)
    for bond in bonds:#isn't this and above step the same thing?
        edge_attr.append(bond_features(bond))
        edge_attr.append(bond_features(bond))

    #print(edge_attr) 
    #Store all the information in a graph
    nodes_info = torch.tensor(node_f, dtype=torch.float32)
    edges_indx = torch.tensor(edge_index,dtype=torch.long)
    edges_info = torch.tensor(edge_attr, dtype=torch.float32)
    # print(nodes_info)
    # print(edges_indx)
    # print(edge_attr)
    graph = Data(x = nodes_info, edge_index = edges_indx, edge_attr = edges_info)

    return graph



##########################
# --- Count features --- #
##########################

def n_atom_features():
    atom = Chem.MolFromSmiles('C').GetAtomWithIdx(0)
    return len(atom_features(atom,bool_id_feat=False,
                  explicit_H=False,
                  use_chirality=False))


def n_bond_features():
    bond = Chem.MolFromSmiles('CC').GetBondWithIdx(0)
    return len(bond_features(bond,use_chirality=False))

def get_dataloader(df, index, target, mol_column, batch_size,y_scaler=None,shuffle=False,drop_last=False):
  #df - data, 
    y_values = df.loc[index, target].values.reshape(-1, 1)
    #print('y_values :',y_values[0:10].ravel().reshape(1,-1))
    #Collecting y values
    if y_scaler != None: # if scaler is given apply
      y = y_scaler.transform(y_values).ravel().astype(np.float32)
    else: # otherwise don't
      y = y_values.ravel().astype(np.float32)
    #print('y_values_s :',y[0:10])

    # to get graphs from mol data
    x = df.loc[index, mol_column].progress_apply(mol2torchdata).tolist()
    #print(x)
    #zip is to iterate over two lists parallely
    for data,y_i in zip(x, y):
        data.y = torch.tensor([y_i], dtype=torch.float32)
        
    data_loader = DataLoader(x, batch_size=batch_size,
                             shuffle=shuffle, drop_last=drop_last)  
    
    
    return data_loader,x