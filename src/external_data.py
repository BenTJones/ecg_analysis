#This file is used to load the external data for the model from Physionet
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from src.preprocessing import preprocess_signal

leads = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
norm = set(["426783006"])
strict = set(["428750005"])
broad = strict.union(set(["429622005", "164934002", "59931005", "164930006", "164931005","55930002"]))

def read_diagnostic_codes(base_path:str):
    header = wfdb.rdheader(str(base_path))
    for c in header.comments:
        if c.lower().startswith('dx'):
            return set(x.strip() for x in c.split(":", 1)[1].split(","))
    return set()

def build_labels(root,postive = broad):
    '''Builds labels for the dataset from the diagnostic codes for every .hea'''
    rows = []
    for hea in Path(root).rglob('*.hea'):
        base = hea.with_suffix('')
        dx = read_diagnostic_codes(base)
        if dx == norm:
            rows.append({'id': str(base.resolve()), 'label': 0})
        elif dx & postive:
            rows.append({'id': str(base.resolve()), 'label': 1})
            
    return pd.DataFrame(rows)
    
def load_ext_data(base:str):
    '''Loads one record, reorders the leads to PTB-XL order and applies the same
    filtering and per-lead normalisation as the original PTB-XL pipeline. Returns an array of shape (12, samples).'''
    signal,meta = wfdb.rdsamp(base)
    names = [n.upper() for n in meta['sig_name']]
    signal = signal[:,[names.index(n) for n in leads]]
    return preprocess_signal(signal,meta['fs'])

def cache_signal(df,out='physionet_cache'):
    '''Preprocesses every record in df and saves one array X.npy plus labels.csv. Skips records that fail to load or are not 12 x 5000.'''
    Path(out).mkdir(parents=True,exist_ok=True)
    total = len(df)
    keep = []
    
    for i, record in enumerate(df['id']):
        file_id = Path(record).name
        cache_path = Path(out) / f'{file_id}.npy'
        if cache_path.exists():
            keep.append(i)
            continue
        try:
            x = load_ext_data(record)
            if x.shape != (12, 5000):
                print(f"Skipping {record}: wrong shape {x.shape}")
                continue
            
            np.save(cache_path,x.astype(np.float32,copy=False))
            keep.append(i)
            
        except Exception as e:
            print(f"Failed to load {record}: {e}")
            continue
        
    df = df.iloc[keep].copy()
    if df.empty:
        raise ValueError('No valid results were loaded')
    df.index = df['id'].map(lambda p: Path(p).name)
    df.index.name = 'ecg_id'
    df.to_csv(Path(out) / 'labels.csv', index_label='ecg_id')
    print(f'Kept {len(df)}/{total} records')
    return df

