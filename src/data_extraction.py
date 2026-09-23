import ast
import pandas as pd
import wfdb
import numpy as np
import matplotlib.pyplot as plt

def load_ecg(ecg_id,df,ecg_path,sampling_rate = 100):
    row = df.loc[ecg_id]
    if sampling_rate == 100:
        record_path = ecg_path + row['filename_lr']
        
    elif sampling_rate == 500:
        record_path = ecg_path + row['filename_hr']
    else:
        raise ValueError('Sampling rate should be 100 or 500')
        
    signal,metadata = wfdb.rdsamp(record_path)
    
    return signal, metadata,row

def get_diagnostic_superclass(scp_codes,scp_states):
    diagnostic_table = scp_states[scp_states["diagnostic"] == 1]
    'Ensures only diagnostic labels are used not minor ecg observations'
    
    classes = []
    for code in scp_codes.keys():
        if code in diagnostic_table.index:
            diagnostic_class = diagnostic_table.loc[code,'diagnostic_class']
            classes.append(diagnostic_class)
            
    return list(set(classes))

def add_superclass_col(df,scp_statements):
    df = df.copy()
    df['diagnostic_superclass'] = df['scp_codes'].apply(
        lambda codes: get_diagnostic_superclass(codes,scp_statements) 
    )
    return df

def make_binary_label(classes):
    '''Creates a map for normal ECG vs ST/T abnormality or anyother diangosis 
        0 for pure norm 
        1 for any STTC'''
    classes = set(classes)
    
    if classes == {'NORM'}:
        return 0
    
    if 'STTC' in classes:
        return 1
    
    return np.nan

def add_binary_col(df):
    df = df.copy()
    df['label'] = df['diagnostic_superclass'].apply(make_binary_label)
    return df

def plot_ecg(signal,meta,title = None):
    '''Needs ECG data already to be read and seperated into signal and metadata to allow for correct plotting
    Shows each lead one by one in a compisite plot'''
    fs = meta['fs']
    lead_name = meta['sig_name']
    t = np.arange(signal.shape[0]) / fs
    
    fig,axes = plt.subplots(
        nrows = 12,
        ncols = 1,
        figsize = (12,15),
        sharex = True
    )
    
    for lead_idx,ax in enumerate(axes):
        ax.plot(t,signal[:,lead_idx])
        ax.set_ylabel(lead_name[lead_idx],rotation = 0)
        ax.grid(True, alpha = 0.25)
        
    axes[-1].set_xlabel("Time (s)")

    if title is not None:
        fig.suptitle(title, y=1.02)

    plt.tight_layout()
    plt.show()