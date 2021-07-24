import pandas as pd
import numpy as np
# try:
#   from rdkit import Chem
#   mol = Chem.MolFromSmiles('O=C2NC(=O)C(c1ccccc1)(C(=O)N2)CC')
#   print(mol)
# except:
#   print('No Chem')
#   print(rdkit.__version__)

def restructDB(path,db_type,num):
  from rdkit import Chem
  #mol = Chem.MolFromSmiles('O=C2NC(=O)C(c1ccccc1)(C(=O)N2)CC')
  #print(mol)
  if db_type == 0 and num==4:
    df = pd.read_excel(path)
    df['SMILES'] = df['SMILES'].str.replace(' ','')
    df['SMILES'] = df['SMILES'].str.replace('\n','')
    print('works!!')
    df['mol'] = df.iloc[1:]['SMILES'].apply(Chem.MolFromSmiles)
    df = df[~df['mol'].isna()]
    print('works!!')
  else:
    df = pd.read_table(path, names=('Id','CAS','SMILES', 'Status','Experimental value', 'Predicted value'))
  # Fill 'mol' column with 'smiles' to 'mol' dat
    df['mol'] = df.iloc[1:]['SMILES'].apply(Chem.MolFromSmiles)
              
  # Remove 'None' values present in the data
  df1 = df.mask(df.astype(object).eq('None')).dropna()
  df1 = df1.reset_index(drop=True) 
  df1['Id'] = [i for i in df1.axes[0]]
      
  # Fill 'Molecular Weight' column. Might come in use during molecular weight splits.
  mol_wt = 'Molecular Weight'
  df1[mol_wt] = df1.iloc[0:]['mol'].apply(Descriptors.MolWt)

  # Convert string data to machine readable data.
  col = 'Experimental value'
  col1= 'Predicted value'
  if db_type == 0 and num != 4:
    df1[col].replace('Carcinogen', 1, inplace=True)
    df1[col].replace('NON-Carcinogen', 0, inplace=True)
    df1['Predicted value'].replace('Carcinogen', 1, inplace=True)
    df1['Predicted value'] = pd.to_numeric(df1['Predicted value'], errors='coerce')
    df1['Predicted value'] = df1['Predicted value'].fillna(0)
    df1 = df1.astype({'Predicted value': np.int64})
  elif db_type == 0 and num == 4:
    df1[col].replace('Carcinogen', 1, inplace=True)
    df1[col].replace('Non-Carcinogen', 0, inplace=True)
  elif db_type == 2:
    df1[col].replace('Mutagenic', 1, inplace=True) #ic
    df1[col].replace('NON-Mutagenic', 0, inplace=True)
    df1['Predicted value'].replace('Mutagenic',1,inplace=True)
    df1['Predicted value'] = pd.to_numeric(df1['Predicted value'], errors='coerce')
    df1['Predicted value'] = df1['Predicted value'].fillna(0)
    df1 = df1.astype({'Predicted value': np.int64})
  # Return the restructured data frame.
  return df1



def link(db_type,num):
    lst1 = ['carcinogenicity','fish_acute_toxicity','mutagenicity','octanol-water_partition']
    car_lst = ['antares/dataset_CARC_ANTARES.txt','caesar/dataset_CARC_CAESAR.txt','iss/dataset_CARC_ISS.txt','isscan-cgx/dataset_CARC_ISSCAN-CGX.txt','Cancer_3D_New.xlsx']
    fat_lst = ['epa/dataset_FATHEAD_EPA.txt','knn/dataset_FISH_KNN.txt','nic/dataset_FISH_NIC.txt'] #,'irfmn/dataset_FISH_IRFMN.txt'
    muta_lst = ['SARPY/dataset_MUTA_SARPY.txt','KNN/dataset_MUTA_KNN.txt','ISS/dataset_MUTA_ISS.txt','CAESAR/dataset_MUTA_CAESAR.txt']
    oct_lst = ['mlogp/dataset_LOGP_MLOGP.txt','meylan/dataset_LOGP_MEYLAN.txt','alogp/dataset_LOGP_ALOGP.txt']
    path =''
    if db_type == 0:
      if num in range (0,len(car_lst)):path =  lst1[0] +'/'+car_lst[num]
    elif db_type == 1:
      if num in range (0,len(fat_lst)):path =  lst1[1] +'/'+fat_lst[num]
    elif db_type == 2:
      if num in range (0,len(muta_lst)):path =  lst1[2] +'/'+muta_lst[num]
    elif db_type == 3:
      if num in range (0,len(oct_lst)):path =  lst1[3] +'/'+oct_lst[num]
    else:
      print('Invalid data base type')

    if not path:
      return 'Path is empty'
    else:
      return '/content/drive/My Drive/GNNs/Property_databases/' + path

def generateIndices(df,Status):
  print('in generateIndices')
  idx_data = np.where(df['Status']==Status)
  # Unravel the tuple
  d_list = [ a for (a)  in idx_data] # List with array
  # Convert to list
  idx_list = list(d_list[0])
  new_df = df.loc[df.index.isin(idx_list)]
  y_data = new_df['Experimental value'].values.reshape(-1,1)
  return idx_list,new_df,y_data