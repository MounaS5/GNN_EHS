
class restructDataFrame:
  lst1    = ['carcinogenicity','fish_acute_toxicity','mutagenicity','octanol-water_partition']
  try:
    import pandas as __pd
    import numpy as __np
    print(__pd.__version__)
  except:
    raise Exception('rdkit, marshal and regex have to be installed.')
   
  def __init__(self,db_type=None,num=None):#,lst1=None
		# # Instance Variable # Change with the object 
    self.db_type = db_type
    self.num     = num

    
  def link(self,db_type,num):
    #lst1 = ['carcinogenicity','fish_acute_toxicity','mutagenicity','octanol-water_partition']
    car_lst = ['antares/dataset_CARC_ANTARES.txt','caesar/dataset_CARC_CAESAR.txt','iss/dataset_CARC_ISS.txt','isscan-cgx/dataset_CARC_ISSCAN-CGX.txt','Cancer_3D_New.xlsx']
    fat_lst = ['epa/dataset_FATHEAD_EPA.txt','knn/dataset_FISH_KNN.txt','nic/dataset_FISH_NIC.txt'] #,'irfmn/dataset_FISH_IRFMN.txt'
    muta_lst = ['SARPY/dataset_MUTA_SARPY.txt','KNN/dataset_MUTA_KNN.txt','ISS/dataset_MUTA_ISS.txt','CAESAR/dataset_MUTA_CAESAR.txt']
    oct_lst = ['mlogp/dataset_LOGP_MLOGP.txt','meylan/dataset_LOGP_MEYLAN.txt','alogp/dataset_LOGP_ALOGP.txt']
    path =''
    if db_type == 0:
      if num in range (0,len(car_lst)):path =  self.lst1[0] +'/'+car_lst[num]
    elif db_type == 1:
      if num in range (0,len(fat_lst)):path =  self.lst1[1] +'/'+fat_lst[num]
    elif db_type == 2:
      if num in range (0,len(muta_lst)):path =  self.lst1[2] +'/'+muta_lst[num]
    elif db_type == 3:
      if num in range (0,len(oct_lst)):path =  self.lst1[3] +'/'+oct_lst[num]
    else:
      print('Invalid data base type')

    if not path:
      return 'Path is empty'
    else:
      return '/content/drive/My Drive/GNNs/Property_databases/' + path
