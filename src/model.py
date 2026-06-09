import torch
import torch.nn as nn 

class simpleECGNN(nn.Module):
    def __init__(self):
        super.__init__()
        
        self.features = nn.Sequential(
            nn.Conv1d(12,32,kernel_size= 7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32,64,kernel_size=7,padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64,128,kernel_size=7,padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),

        )
        
        self.classifier = nn.Linear(128,1)
        
    def forward(self, x):
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.classifier(x)
        x = x.squeeze(1)
        return x
        