import torch
from torch.utils.data import Dataset
from numpy import loadtxt, float32, int32
class AwA2ResNetDataset(Dataset):
    def __init__(self, feature_file, label_file):
        self.features = torch.from_numpy(loadtxt(feature_file, dtype=float32))
        self.labels = torch.from_numpy(loadtxt(label_file, dtype=int32))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'labels': self.labels[idx]}