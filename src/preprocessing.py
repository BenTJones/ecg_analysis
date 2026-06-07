from src.data_extraction import load_ecg

import numpy as np
from scipy.signal import butter, sosfiltfilt

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

def load_and_preprocess(id,df,path,sampling_rate=500,use_filter = True, low=0.5,high=40):
    signal,meta,row = load_ecg(
        id,
        df,
        path,
        sampling_rate
    )
    signal = to_channels_first(signal)
    fs = meta['fs']
    
    if use_filter:
        signal = bandpass_filter(signal,fs,low,high)
    
    signal = bylead_normalisation(signal)
    
    return signal,meta,row

