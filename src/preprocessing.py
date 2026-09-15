from src.data_extraction import load_ecg

import numpy as np
from scipy.signal import butter, sosfiltfilt
import os

def to_channels_first(signal):
    '''Signal read in the wrong shape for Torch as a tensor, so need to be flipped'''
    return signal.T.astype(np.float32)

def bandpass_filter(signal,fs,low_cut = 0.5, high_cut= 50, order = 4):
    nyquist = fs / 2 #Needed to correctly apply scipy filters
    
    low = low_cut /nyquist
    high = high_cut / nyquist
    
    out = butter(order,[low,high], btype= 'band', output ='sos')
    filtered = sosfiltfilt(out,signal,axis=1)
    return filtered.astype(np.float32)

def bylead_normalisation(signal):
    epsilon = 1e-8
    mean = signal.mean(axis=1,keepdims=True)
    std = signal.std(axis=1,keepdims = True)
    
    normalised = (signal - mean) / (std + epsilon)
    return normalised.astype(np.float32)

def preprocess_signal(signal, fs, use_filter=True, low=0.5, high=40):
    signal = to_channels_first(signal)
    if use_filter:
        signal = bandpass_filter(signal, fs, low, high)
    signal = bylead_normalisation(signal)
    return signal

def load_and_preprocess(id,df,path,sampling_rate=500,use_filter = True, low=0.5,high=40):
    signal,meta,row = load_ecg(
        id,
        df,
        path,
        sampling_rate
    )
    fs = meta['fs']
    signal = preprocess_signal(signal, fs, use_filter=use_filter, low=low, high=high)

    return signal,meta,row

def preprocess_and_cache(df,data_path = 'data/', sampling_rate = 500, use_filt = True,low = 0.5, high = 40):
    SAVE_DIR = "data/preprocessed_500"
    os.makedirs(SAVE_DIR, exist_ok=True)
    for i,ecg_id in enumerate(df.index):
        save_path = os.path.join(SAVE_DIR,f'{ecg_id}.npy')
        
        if os.path.exists(save_path):
            continue
        
        signal,meta,row = load_and_preprocess(
            ecg_id, 
            df,
            data_path,
            sampling_rate,
            use_filt,
            low,
            high
        )
        
        signal = signal.astype(np.float32)
        np.save(save_path, signal)
