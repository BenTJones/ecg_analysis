import matplotlib.pyplot as plt
from src.data_extraction import load_ecg, plot_ecg


def loss_curve(history,save_path = 'results/plots/loss_curve.png'):
    plt.figure(figsize=(9,5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve (Training and Validation)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    
def val_metrics(history,save_path = 'results/plots/val_metrics.png'):
    plt.figure(figsize=(9,5))
    plt.plot(history['val_auroc'], label='Validation AUROC')
    plt.plot(history['val_auprc'], label='Validation AUPRC')
    plt.plot(history['val_f1'], label='Validation F1 Score')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    
def plot_ecg_by_id(ecg_id,df,path = 'data/'):
    signal,meta,row = load_ecg(ecg_id,df,path)
    plot_ecg(signal,meta,title = f"ECG ID: {ecg_id}")
    