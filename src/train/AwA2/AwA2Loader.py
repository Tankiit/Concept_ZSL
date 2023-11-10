import torch
from torch.utils.data import Dataset, random_split
from numpy import loadtxt, float32
class AwA2ResNetDataset(Dataset):
    def __init__(self, feature_file, label_file, file_paths_file):
        self.features = torch.from_numpy(loadtxt(feature_file, dtype=float32))
        self.labels = torch.from_numpy(loadtxt(label_file, dtype=int))
        self.file_paths = open(file_paths_file, "r").read().split("\n")[:-1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'labels': self.labels[idx] - 1}

import os
import pandas as pd
from torchvision.datasets.folder import default_loader
class AwA2Dataset(Dataset):
    def __init__(self, root, transform=None, exclude=[], loader=default_loader, skip_first_n=0, end_at_n=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.exclude = exclude

        self.classes = [d.name for d in os.scandir(self.root) if d.is_dir() and d.name not in self.exclude]

        self.data = pd.DataFrame(columns=['image', 'label'])
        for i, c in enumerate(self.classes):
            # get all files in the directory (full path)
            files = [os.path.join(self.root, c, f) for f in os.listdir(os.path.join(self.root, c))][skip_first_n:]
            
            if end_at_n is not None:
                files = files[:end_at_n]
            
            self.data = self.data._append(pd.DataFrame({'image': files, 'label': i}), ignore_index=True)
            
    def set_transform(self, new_transform):
        self.transform = new_transform
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data.iloc[index]['image']
        label = self.data.iloc[index]['label']
        img = self.loader(path)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return {"images": img, "labels": label}
    
def get_all_classes(root):
    return [d.name for d in os.scandir(root) if d.is_dir()]
    
if __name__ == '__main__':
    exclude = ["sheep", "dolphin", "bat", "seal", "blue+whale", "rat", "horse", "walrus", "giraffe", "bobcat"]
    dataset = AwA2Dataset(root='/storage/Animals_with_Attributes2/JPEGImages', exclude=[])
    print(len(dataset))

    train_length = int(len(dataset) * 0.8)
    val_length = len(dataset) - train_length
    train_set, val_set = random_split(dataset, [train_length, val_length])
    print(len(train_set), len(val_set))