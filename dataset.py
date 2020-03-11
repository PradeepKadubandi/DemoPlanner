from torch.utils.data import Dataset
import torch
import numpy as np

class NumpyCsvDataSet(Dataset):
    def __init__(self, csv_file, device=None):
        self.csv_file = csv_file
        self.data = torch.as_tensor(np.loadtxt(csv_file, delimiter=',', dtype=float)).to(device)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]