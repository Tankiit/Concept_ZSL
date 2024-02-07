import os
import pandas as pd
from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'

    def __init__(self, root, train=True, transform=None, all_data=False, train_split_file=None, loader=default_loader, exclude=[]):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.exclude = exclude
        self.all_data = all_data
        if train_split_file == None:
            self.train_split_file = os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt')
        else:
            self.train_split_file = train_split_file

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(self.train_split_file,
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')
        
        self.data = self.data[~self.data['target'].isin(self.exclude)]
        
        self.data['target'] = pd.factorize(self.data['target'])[0]
        
        if not self.all_data:
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

from random import sample
from collections import Counter
def make_train_split_file(images_per_class, output_file, root):
    # creates a train split file with the given number of images per class in the training set
    # the rest of the images are in the test set
    # train_test_split.txt is of the form: <image_id> <is_training_image>\n<image_id> <is_training_image>\n...

    data = pd.read_csv(os.path.join(root, 'CUB_200_2011', 'image_class_labels.txt'), sep=' ', names=['img_id', 'target'])
    
    # count how many elements in each class and randomly sample
    counts = dict(Counter(data['target']))
    train_split = {}
    curr_index = 1
    for key in counts.keys():
        train_split[key] = [x+curr_index for x in sample(range(counts[key]), images_per_class)]
        curr_index += counts[key]

    # create the train_test_split.txt file
    with open(output_file, "w") as f:
        for _, row in data.iterrows():
            if row['img_id'] in train_split[row['target']]:
                f.write(str(row['img_id']) + " 1\n")
            else:
                f.write(str(row['img_id']) + " 0\n")
    
import numpy as np
def make_ZSL_sets(root, NUM_EXCLUDE, train_transform, val_transform):
    indices = (np.random.permutation(200)+1).tolist()
    
    ZSL_train_set = Cub2011(root, transform=train_transform, exclude = indices[NUM_EXCLUDE:])
    train_set = Cub2011(root, transform=train_transform, exclude = indices[:NUM_EXCLUDE])
    
    ZSL_test_set = Cub2011(root, transform=val_transform, train=False, exclude = indices[NUM_EXCLUDE:])
    test_set = Cub2011(root, transform=val_transform, train=False, exclude = indices[:NUM_EXCLUDE])
    
    return train_set, test_set, ZSL_train_set, ZSL_test_set
    
if __name__ == "__main__":
    indices = (np.random.permutation(200)+1).tolist()
    
    selected = indices[20:]
    print(min(selected), max(selected))
    
    root = "datasets/"
    dset = Cub2011(root, exclude = selected)
