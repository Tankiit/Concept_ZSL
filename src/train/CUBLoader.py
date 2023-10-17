import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'

    def __init__(self, root, train=True, transform=None, loader=default_loader, exclude=[]):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.exclude = exclude

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        
        self.data = self.data[~self.data['target'].isin(self.exclude)]
        
        self.data['target'] = pd.factorize(self.data['target'])[0]
        
        
        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
            
    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception as e:
            print(e)
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)
            
        return {"images": img, "labels": target}

import numpy as np
def make_ZSL_sets(NUM_EXCLUDE, train_transform, val_transform):
    indices = (np.random.permutation(200)+1).tolist()
    
    ZSL_train_set = Cub2011("/storage/CUB", transform=val_transform, exclude = indices[NUM_EXCLUDE:])
    train_set = Cub2011("/storage/CUB", transform=train_transform, exclude = indices[:NUM_EXCLUDE])
    
    ZSL_test_set = Cub2011("/storage/CUB", transform=val_transform, train=False, exclude = indices[NUM_EXCLUDE:])
    test_set = Cub2011("/storage/CUB", transform=train_transform, train=False, exclude = indices[:NUM_EXCLUDE])
    
    return train_set, test_set, ZSL_train_set, ZSL_test_set
    
if __name__ == "__main__":
    indices = (np.random.permutation(200)+1).tolist()
    
    selected = indices[20:]
    print(min(selected), max(selected))
    
    dset = Cub2011("/storage/CUB", exclude = selected)
