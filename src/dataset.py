import torch
from torch.utils.data import Dataset
from src.preprocessing import load_and_preprocess
import numpy as np
import os

class ECGdataset(Dataset):
    def __init__(
        self,
        df,
        path = 'data/',
        samp_rate = 500,
        use_filt = True
    ):
        '''
        Torch style dataset for the ECG classifcation task
        '''
        
        self.df = df.copy()
        self.path = path
        self.samp_rate = samp_rate
        self.use_filt = use_filt
        
        self.ids = self.df.index.tolist()
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        id = self.ids[index]
        
        signal,meta,row = load_and_preprocess(
            id,
            self.df,
            self.path,
            self.samp_rate,
            self.use_filt
        )
        
        label = row['label']
        signal = torch.tensor(signal, dtype = torch.float32)
        label = torch.tensor(label, dtype = torch.float32)
        
        return signal, label
    


class CachedECGDataset(Dataset):
    def __init__(self,df,cache_dir = 'data/preprocessed_500'):
        self.df = df.copy()
        self.cache_dir = cache_dir
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,index):
        row = self.df.iloc[index]
        ecg_id = row.name
        signal_path = os.path.join(self.cache_dir,f'{ecg_id}.npy')
        signal = np.load(signal_path)
        label = np.float32(row['label'])
        
        return torch.tensor(signal, dtype = torch.float32), torch.tensor(label, dtype = torch.float32)
    

        