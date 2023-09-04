import torch
import os
import random
import argparse
import datetime
import json


from utils import get_dataset


parser = argparse.ArgumentParser()
#parser.add_argument("--concept-bank", required=True, type=str, help="Path to the concept bank")
#parser.add_argument("--out-dir", required=True, type=str, help="Output folder for model/run info.")
parser.add_argument("--dataset", default="cub", type=str)
parser.add_argument("--backbone-name", default="resnet18_cub", type=str)
parser.add_argument("--device", default="cuda", type=str)
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--alpha", default=0.99, type=float, help="Sparsity coefficient for elastic net.")
parser.add_argument("--lam", default=1e-5, type=float, help="Regularization strength.")
parser.add_argument("--lr", default=1e-3, type=float)

args=parser.parse_args()


train_loader,test_loader,idx_to_class,classes=get_dataset(args)




