import os

from captum.attr import Saliency
import torch
import numpy as np
import matplotlib.pyplot as plt

leads = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CHEST_LEAD_IDXS = list(range(6, 12))

def get_ecg_from_dataset(ecg_id,dataset,df):
    if ecg_id not in df.index:
        raise ValueError(f"ECG ID {ecg_id} not found in dataframe index.")
    
    idx = df.index.get_loc(ecg_id)
    x,y = dataset[idx]
    return x,y,idx

def compute_saliency(model, x, device):
    """Compute input-gradient saliency for a single ECG tensor."""
    model.eval()
    x = x.unsqueeze(0).to(device).float()
    x.requires_grad = True
    saliency = Saliency(model)

    attr = saliency.attribute(x)
    signal = x.squeeze(0).detach().cpu().numpy()
    sal_vals = attr.squeeze(0).detach().cpu().numpy()
    return signal, sal_vals

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
    x, y, _ = get_ecg_from_dataset(ecg_id, dataset, df)
    signal, sal_vals = compute_saliency(model, x, device)

    with torch.no_grad():
        prob = torch.sigmoid(model(x.unsqueeze(0).to(device).float())).item()

    sal = np.abs(sal_vals[lead_idx])
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
        alpha=0.75,
        cmap="Reds",
    )

    plt.colorbar(points, label="Saliency")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title(
        f"Saliency map - ECG {ecg_id} - Lead {lead_name} - "
        f"True: {int(y.item())} | Pred prob: {prob:.3f}"
    )
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_saliency_for_leads(
    model,
    dataset,
    df,
    ecg_id,
    device,
    lead_indices,
    sampling_rate=500,
    save_dir=None,
    case_prefix="",
):
    """Plot saliency maps for multiple leads on one ECG."""
    for lead_idx in lead_indices:
        save_path = None
        if save_dir is not None:
            lead_name = leads[lead_idx]
            save_path = os.path.join(
                save_dir,
                f"saliency_{case_prefix}_{ecg_id}_lead{lead_name}.png",
            )
        plot_saliency_for_lead(
            model=model,
            dataset=dataset,
            df=df,
            ecg_id=ecg_id,
            device=device,
            lead_idx=lead_idx,
            sampling_rate=sampling_rate,
            save_path=save_path,
        )

