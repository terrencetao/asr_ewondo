# 3 steps 
# - random choice of phrase

import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os
def format_dataset(data_folder,
    save_folder, test_per, dev_per,nb_loc,seed, dev=False):
    """
    This class prepare the dataset.
    

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original  dataset is stored.
    save_folder : str
        The directory where to store the csv files
    """
    
    data = pd.read_csv(os.path.join(data_folder,'transcription.csv'),dtype={'ID': 'str'})

    train, test = train_test_split(data, test_size=test_per, random_state =seed)
    
    if dev :
    	test, dev = train_test_split(test, test_size=dev_per, random_state=seed)
    else:
    	dev = test

    if os.path.exists(save_folder):
    	shutil.rmtree(save_folder, ignore_errors=True)

    os.makedirs(save_folder)

    train_folder =  os.path.join(save_folder,'train')
    test_folder =   os.path.join(save_folder, 'test')
    dev_folder =  os.path.join(save_folder, 'dev')


    create_folder(train,train_folder,data_folder,nb_loc)
    create_folder(test,test_folder,data_folder,nb_loc)
    create_folder(dev,dev_folder,data_folder,nb_loc)

def create_folder(df,save_folder,data_folder,nb_loc):
    os.makedirs(save_folder)
    df[['ID','Ewondo']].astype({'ID': 'str'}).to_csv(os.path.join(save_folder,"transcription.txt"),header=False, index=False, sep = " ")
    for i in range(1,nb_loc): 
        if i<10:
            loc = 'loc_00'+ str(i)
        elif i<100:
            loc = 'loc_0' +str(i)
        else:
            loc = 'loc_'+ str(i)
        os.path.join(save_folder,loc)
        os.makedirs(os.path.join(save_folder,loc))
        for idx in list(df['ID']):
            wav = 'STE-'+ idx + '.wav'
            try:
                shutil.copyfile(os.path.join(os.path.join(data_folder,loc), wav),os.path.join(os.path.join(save_folder,loc), wav) )
            except:
                pass

   

if __name__ == "__main__":
   format_datatest(data_folder='Corpus Phrases simples',
    save_folder='datasets', test_per=0.10, dev_per=0.10,nb_loc=5, dev=False, seed=45)