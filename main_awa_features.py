import numpy as np
import torch
import  torch.nn as nn
import torch.nn.functional as F


import os
import glob


import clip
import pandas as pd
import argparse

import pdb
import timm

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

from skorch import NeuralNetClassifier



class SimpleMLP(torch.nn.Module):
      def __init__(self,input_dim=2048,hidden_dim=4000,n_classes=50):
          super(SimpleMLP,self).__init__()
          self.linear=nn.Linear(input_dim,hidden_dim)
          #self.nonlin=nn.ReLU()
          self.dropout=nn.Dropout(0.5)
          self.output=nn.Linear(hidden_dim,n_classes)
      def forward(self,x):
          X=self.linear(x)
          X=F.relu(X)
          X=self.dropout(X)
          X=F.softmax(self.output(X),dim=-1)
          return X




if __name__=="__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument('--dataset', type=str, default="AwA", help='Dataset')
   parser.add_argument('--concepts', type=str, default="True", help='if concepts are available')
   args=parser.parse_args()
   if args.concepts=='True':

      concepts=pd.read_csv('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/predicates.txt',sep="\t",header=None)[1].to_list()

   device = "cuda" if torch.cuda.is_available() else "cpu"
   #model, preprocess = clip.load("ViT-B/32", device=device)
   #concepts=clip.tokenize(concepts).to(device)
   classes=pd.read_csv('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Animals_with_Attributes2/classes.txt',sep='\t',header=None)[1].to_list()
   features_path='/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Features/ResNet101'
   features_file=np.loadtxt('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt')
   filenames=pd.read_csv('/media/steven/WDD/research/data/Concept_ZSL/Animals_with_Attributes2/Features/ResNet101/AwA2-filenames.txt',header=None)
   index_classes=[]
   
   for i,j in zip(filenames[0],features_file):
       index_classes.append(classes.index(i.strip().split('_')[0]))
   index_classes=np.array(index_classes).astype(np.int64)    
   features_file=features_file.astype(np.float32)
   X_train,X_test,Y_train,Y_test=train_test_split(features_file,index_classes,test_size=0.2,shuffle=True)
   #svms = SVC(gamma=0.001)
   #svms.fit(X_train,Y_train)
   #predicted=svms.predict(X_test)

   #print(
   # f"Classification report for classifier {clf}:\n"
   # f"{metrics.classification_report(Y_test, predicted)}\n"
   #)
   net=NeuralNetClassifier(SimpleMLP,max_epochs=20,lr=0.001,device='cuda')

   net.fit(X_train,Y_train)
   predicted=net.predict(X_test)
   print(
    f"Classification report for classifier {net}:\n"
    f"{metrics.classification_report(Y_test, predicted)}\n"
   )


    
              
    
   
   
