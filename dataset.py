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

class DataObject:
    def __init__(self, x_dim, y_dim, u_dim, I_dim):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.u_dim = u_dim
        self.I_dim = I_dim

        self.y_end = self.x_dim+self.y_dim
        self.u_end = self.y_end+self.u_dim

    def fill_from_array(self, input):
        self.x = input[:, 0:self.x_dim]
        self.y = input[:, self.x_dim:self.y_end]
        self.u = input[:, self.y_end:self.u_end]
        self.I = input[:, self.u_end:]
    
    def to_array(self):
        return np.concatenate((self.x, self.y, self.u, self.I), axis=1)

    def from_values(x, y, u, I):
        val = DataObject(x.shape[0], y.shape[0], u.shape[0], I.shape[0])
        val.x = x
        val.y = y
        val.u = u
        val.I = I
        return val