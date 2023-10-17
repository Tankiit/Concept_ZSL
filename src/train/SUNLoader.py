import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset
import scipy.io
import numpy as np

class SUNAttr(Dataset):
    root = "/storage/SUN"

    def __init__(self, train=True, transform=None, loader=default_loader, indices=[]):
        self.transform = transform
        self.loader = default_loader
        self.train = train
        
        images = scipy.io.loadmat(os.path.join(self.root, "SUNAttributeDB/images.mat"))
        im_list = [list(images['images'][i][0])[0] for i in range(len(images['images']))]
        self.imgs = [i for i in im_list]

        attributes = scipy.io.loadmat(os.path.join(self.root, "SUNAttributeDB/attributeLabels_continuous.mat"))
        self.labels = np.argmax(attributes['labels_cv'], axis=1)
        
        self.data = pd.DataFrame({'filepath': self.imgs, 'target': self.labels})
        
        if indices:
            self.data = self.data.iloc[indices]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, "images", sample.filepath)
        target = sample.target
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
            
        return {"images": img, "labels": target}

def make_sets(split_size, train_transform, val_transform):
    base_set = SUNAttr()
    indices = np.random.permutation(len(base_set)).tolist()
    train_size = int(len(indices)*(1-split_size))
    test_size = int(len(indices)*split_size)
    
    train_set = SUNAttr(transform=train_transform, indices = indices[:train_size])
    test_set = SUNAttr(transform=val_transform, indices = indices[-test_size:])
    
    return train_set, test_set
    
if __name__ == "__main__":
    SUN = SUNAttr()