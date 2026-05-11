import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MyDS(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.input = None
        self.label = None

        self.process(data, label)
    
    def __getitem__(self, index):
        return self.input[index], self.label[index]
    
    def __len__(self):
        return len(self.input)

    def process(self, data, label):
        self.input = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype = torch.float32)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, out_dim, dropout = 0.2):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(hidden_dim, out_dim)
        # self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.mlp(x)
        x = self.fc(x)
        return x
