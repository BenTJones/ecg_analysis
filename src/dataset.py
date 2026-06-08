import torch
from torch.utils.data import Dataset
from src.preprocessing import load_and_preprocess

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
    
