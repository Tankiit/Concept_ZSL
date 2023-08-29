import numpy as np
import torch

import os
import glob


import clip
import pandas as pd
import argparse

import pdb
import timm

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


if __name__=="__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--dataset', type=str, default="AwA", help='Dataset')
   parser.add_argument('--concepts', type=str, default="True", help='if concepts are available')


   args=parser.parse_args()
   if args.concepts=='True':

      concepts=pd.read_csv('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/predicates.txt',sep="\t",header=None)[1].to_list()

   device = "cuda" if torch.cuda.is_available() else "cpu" 
   model, preprocess = clip.load("ViT-B/32", device=device)
   concepts=clip.tokenize(concepts).to(device)
   train_classes=pd.read_csv('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Animals_with_Attributes2/trainclasses.txt',header=None)[0].to_list()
   features_path='/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Features/ResNet101'
   features_file=np.loadtxt('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt')
   filenames=pd.read_csv('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Features/ResNet101/AwA2-filenames.txt',header=None)
   index_classes=[]
   for i,j in enumerate(filenames[0].to_list()):
       for k in train_classes:
           if k in j:
              index_classes.append(i)
           else:
                pass
    
   indexes=np.array(index_classes)
   #pdb.set_trace()
   train_feats=features_file[indexes]
   
   #with torch.no_grad():
   #     concept_features = model.encode_text(concepts)
        
        
              
