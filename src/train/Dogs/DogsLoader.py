import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, random_split

class DogsDataset(Dataset):
    def __init__(self, root, transform=None, exclude=[], loader=default_loader):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.exclude = exclude

        self.classes = [d.name for d in os.scandir(self.root) if d.is_dir() and d.name not in self.exclude]

        self.data = pd.DataFrame(columns=['image', 'label'])
        for i, c in enumerate(self.classes):
            # get all files in the directory (full path)
            files = [os.path.join(self.root, c, f) for f in os.listdir(os.path.join(self.root, c))]
            self.data = self.data._append(pd.DataFrame({'image': files, 'label': i}), ignore_index=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data.iloc[index]['image']
        label = self.data.iloc[index]['label']
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
    
def get_all_classes(root):
    return [d.name for d in os.scandir(root) if d.is_dir()]

if __name__ == '__main__':
    dataset = DogsDataset(root='/storage/Dogs/Images', exclude=[])
    print(len(dataset))

    train_legth = int(len(dataset) * 0.8)
    val_length = len(dataset) - train_legth
    train_set, val_set = random_split(dataset, [train_legth, val_length])
    print(len(train_set), len(val_set))