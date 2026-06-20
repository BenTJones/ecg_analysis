from captum.attr import Saliency
import torch
import numpy as np
import matplotlib.pyplot as plt

leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

def get_ecg_from_dataset(ecg_id,dataset,df):
    if ecg_id not in df.index:
        raise ValueError(f"ECG ID {ecg_id} not found in dataframe index.")
    
    idx = df.index.get_loc(ecg_id)
    x,y = dataset[idx]
    return x,y,idx

def compute_saliency(model,x,device):
    '''Compute the saliency for a single ECG tensor'''
    model.eval()
    x = x.unsqueeze(0).to(device).float()
    x.requires_grad = True
    saliency = Saliency(model)
    
    attr = saliency.attribute(x)
    signal = x.squeeze(0).detach().cpu().numpy()
    sal_vals = attr.squeeze(0).detach().cpu().numpy()
    #Basically Extracts raw gradient model calculates gradient wise
    return signal,sal_vals

def plot_saliency_for_lead(
    model,
    dataset,
    df,
    ecg_id,
    device,
    lead_idx=1,
    sampling_rate=500,
    save_path=None,
):
    x,y,idx = get_ecg_from_dataset(ecg_id,dataset,df)
    signal,sal_vals = compute_saliency(model,x,device)
    
    sal = np.abs(sal_vals[lead_idx]) #From equation of abs of gradient
    sal = sal / (sal.max() + 1e-8) 
    t = np.arange(signal.shape[1]) / sampling_rate
    
    lead_name = leads[lead_idx]
    plt.figure(figsize=(14, 4))
    plt.plot(t, signal[lead_idx], linewidth=1.2, label=f"Lead {lead_name}")
    
    points = plt.scatter(
        t,
        signal[lead_idx],
        c=sal,
        s=5,
        alpha=0.8,
    )

    plt.colorbar(points, label="Saliency")
    plt.xlabel("Time (s)")
    plt.ylabel("Normalised amplitude")
    plt.title(f"Saliency map | ECG ID {ecg_id} | Lead {lead_name} | True label: {int(y.item())}")
    plt.legend()
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()