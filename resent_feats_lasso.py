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

from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from sklearn.model_selection import train_test_split
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
   train_classes=pd.read_csv('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/all_classes.txt',header=None)[0].to_list()

   features_path='/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Features/ResNet101'
   features_file=np.loadtxt('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt')
   filenames=pd.read_csv('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Features/ResNet101/AwA2-filenames.txt',header=None)
   index_classes=[]
   class_labels=[]
   for i,j in enumerate(filenames[0].to_list()):
       for k in train_classes:
           if k in j:
           class_labels.append(train_classes.index(k))

           else:
                pass

   indexes=np.array(index_classes)
   train_feats=features_file[indexes]
   X_train, X_test, y_train, y_test = train_test_split(
   train_feats, class_labels, test_size=0.33, random_state=42)
   from sklearn import linear_model
   reg = linear_model.Lasso(alpha=0.1)
   reg.fit(X_train,y_train)
   reg.predict(X_test)
   print (reg.score(X_test,y_test))
