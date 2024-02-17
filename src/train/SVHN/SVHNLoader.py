import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, random_split

class SVHNDataset(Dataset):
    def __init__(self, root, transform=None, loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader

        self.files = [os.path.join(self.root, d.name) for d in os.scandir(self.root) if d.name.endswith(".png")]
        self.data = pd.DataFrame(self.files, columns =['image'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data.iloc[index]['image']
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return {"images": img}

if __name__ == '__main__':
    dataset = SVHNDataset(root='datasets/SVHN/train')
    print(len(dataset))

    train_length = int(len(dataset) * 0.8)
    val_length = len(dataset) - train_length
    train_set, val_set = random_split(dataset, [train_length, val_length])
    print(len(train_set), len(val_set))