import torch
from torch.utils.data import Dataset

import pandas as pd
from PIL import Image

import os
from glob import glob

class SimpleDataLoader(Dataset):
      def __init__(self,transform=None):
          self.image_path='/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Animals_with_Attributes2/'
          self.transform=transform
          class_to_index = dict()
          # Build dictionary of indices to classes
          with open(self.image_path+'/classes.txt') as f:
               index = 0
               for line in f:
                   class_name = line.split('\t')[1].strip()
                   class_to_index[class_name] = index
                   index += 1
          self.class_to_index = class_to_index

          #with open('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Animals_with_Attributes2/trainclasses.txt'):
          class_names=pd.read_csv('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Animals_with_Attributes2/trainclasses.txt',header=None)[0].to_list()
          
          img_names=[]
          img_index=[]
     
          for line in class_names:
              classes=line.strip()
              FOLDER_DIR=os.path.join(self.image_path+'JPEGImages',classes)
              file_descriptor = os.path.join(FOLDER_DIR, '*.jpg')
              files = glob(file_descriptor)
              class_index = class_to_index[class_name]
              for file_name in files:
                  img_names.append(file_name)
                  img_index.append(class_index)
      def __getitem__(self, index):
          im = Image.open(self.img_names[index])
          if im.getbands()[0] == 'L':
             im = im.convert('RGB')
          if self.transform:
             im = self.transform(im)
          if im.shape != (3,224,224):
             print(self.img_names[index])

          im_index = self.img_index[index]
          im_predicate = self.predicate_binary_mat[im_index,:]
          return im, im_predicate, self.img_names[index], im_index


      def __len__(self):
          return len(self.img_names)

    
