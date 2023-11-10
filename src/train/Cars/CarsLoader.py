import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset, random_split

class CarsDataset(Dataset):
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
        return {"images": img, "labels": label}
    
class CarsZSLDataset(Dataset):
    def __init__(self, root, transform=None, exclude=[], loader=default_loader, skip_first_n=0, end_at_n=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = loader
        self.exclude = exclude

        self.classes = [d.name for d in os.scandir(os.path.join(self.root, "train_images")) if d.is_dir() and d.name not in self.exclude]

        self.data = pd.DataFrame(columns=['image', 'label'])
        for i, c in enumerate(self.classes):
            # get all files in the directory (full path)
            train_files = [os.path.join(os.path.join(self.root, "train_images"), c, f) for f in os.listdir(os.path.join(os.path.join(self.root, "train_images"), c))][skip_first_n:]
            
            if end_at_n is not None:
                train_files = train_files[:end_at_n]
            
            self.data = self.data._append(pd.DataFrame({'image': train_files, 'label': i}), ignore_index=True)
            
            test_files = [os.path.join(os.path.join(self.root, "test_images"), c, f) for f in os.listdir(os.path.join(os.path.join(self.root, "test_images"), c))][skip_first_n:]
            
            if end_at_n is not None:
                test_files = test_files[:end_at_n]
            
            self.data = self.data._append(pd.DataFrame({'image': test_files, 'label': i}), ignore_index=True)

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
    dataset = CarsZSLDataset(root='/storage/Cars', exclude=[])
    print(len(dataset))

    train_length = int(len(dataset) * 0.8)
    val_length = len(dataset) - train_length
    train_set, val_set = random_split(dataset, [train_length, val_length])
    print(len(train_set), len(val_set))