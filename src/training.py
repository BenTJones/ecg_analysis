import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix,accuracy_score

def train_an_epoch(model,loader,optimizer,criterion,device):
    model.train()
    running_loss = 0.0
    
    for idx,(signals, labels) in enumerate(loader):
        signals = signals.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(signals)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * signals.size(0)
        
        #if idx % 20 == 0:
        #    print(
        #        f"Batch {idx}/{len(loader)} - Loss: {loss.item():.4f}",
        #        flush=True
        #    ) USEFUL FOR DEBUGGING BUT CLOGS OUTPUT
    
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss

def eval(model,loader,criterion,device,threshold = 0.5):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for signals,labels in loader:
            signals = signals.to(device)
            labels = labels.to(device)
            logits = model(signals)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            running_loss += loss.item() * signals.size(0)
            
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    all_predictions = (all_probabilities >= threshold).astype(int)
    
    metrics = {
        "loss": running_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "auroc": roc_auc_score(all_labels, all_probabilities),
        "auprc": average_precision_score(all_labels, all_probabilities),
        "f1": f1_score(all_labels, all_predictions),
        "confusion_matrix": confusion_matrix(all_labels, all_predictions),
    }
    
    return metrics, all_labels, all_probabilities, all_predictions

def train(
    model,
    optimizer,
    criterion,
    train_loader,
    val_loader,
    device,
    num_epochs = 20,
    threshold = 0.5,
    patience = 5,
    tolerance = 1e-4
):
    best_val_auroc = 0.0
    best_model_path = 'results/models/best_simple_cnn.pt'
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_auroc': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_auprc': [],
        'val_f1': []
    }
    
    for epoch in range(1,num_epochs+1):
        train_loss = train_an_epoch(model,train_loader,optimizer,criterion,device)
        
        val_metrics, val_labels, val_probs, val_preds = eval(model,val_loader,criterion,device,threshold)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_auroc'].append(val_metrics['auroc'])
        history['val_accuracy'].append(val_metrics['accuracy'])
        history['val_auprc'].append(val_metrics['auprc'])
        history['val_f1'].append(val_metrics['f1'])
        
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_metrics['loss']:.4f} - Val AUROC: {val_metrics['auroc']:.4f} - Val AUPRC: {val_metrics['auprc']:.4f} - Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['auroc'] > (best_val_auroc + tolerance):
            best_val_auroc = val_metrics['auroc']
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with AUROC: {best_val_auroc:.4f}")
            
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break    
        
    return history
    
