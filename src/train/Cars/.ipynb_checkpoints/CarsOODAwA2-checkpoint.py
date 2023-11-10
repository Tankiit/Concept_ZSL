import torchvision.transforms as transforms
from torch.utils.data import Subset
import torch

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

from torchmetrics import Accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

NUM_CLASSES = 196
NUM_FEATURES = 80
accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/models")
from DeiT3AutoPredicates import ResExtr

model = ResExtr(NUM_FEATURES, NUM_CLASSES).to(device)
model.load_state_dict(torch.load("CarsDeiT3Null.pt"))

model.eval()

root_dir = "/storage/Cars"

from CarsLoader import CarsZSLDataset
    
val_dataset = CarsZSLDataset(root=root_dir, transform=val_transform)
validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4)

running_out_attributes = 0.0
running_acc = 0.0

out_count_Cars = []

from tqdm import tqdm
with torch.no_grad():
    for i, vdata in tqdm(enumerate(validation_loader)):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs, vcommit_loss, predicate_matrix = model(vinputs)
        voutputs = voutputs.view(-1, 1, NUM_FEATURES)
        ANDed = voutputs * predicate_matrix
        diff = ANDed - voutputs
        running_acc += accuracy(diff.sum(dim=2), vlabels)
        voutputs = voutputs.view(-1, NUM_FEATURES)
        out_count_Cars.append(voutputs.sum(dim=1))
        running_out_attributes += voutputs.sum() / voutputs.shape[0]
        
avg_oa = running_out_attributes / (i + 1)
avg_acc = running_acc / (i + 1)
print(f"OA: {avg_oa}, ACC: {avg_acc}")

# =============================================================================================

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/AwA2")
from AwA2Loader import AwA2Dataset
val_dataset = AwA2Dataset(root='/storage/Animals_with_Attributes2/JPEGImages', transform=val_transform)

validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4)

running_out_attributes = 0.0

out_count_AwA2 = []

with torch.no_grad():
    for i, vdata in tqdm(enumerate(validation_loader)):
        vinputs = vdata["images"]
        vinputs = vinputs.to(device)
        voutputs, vcommit_loss, predicate_matrix = model(vinputs)
        voutputs = voutputs.view(-1, NUM_FEATURES)
        out_count_AwA2.append(voutputs.sum(dim=1))
        running_out_attributes += voutputs.sum() / voutputs.shape[0]

avg_oa = running_out_attributes / (i + 1)
print(f"OA: {avg_oa}")

# =============================================================================================

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/SVHN")
from SVHNLoader import SVHNDataset
val_dataset = SVHNDataset(root='/storage/SVHN', transform=val_transform)

validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4)

running_out_attributes = 0.0

out_count_SVHN = []

with torch.no_grad():
    for i, vdata in tqdm(enumerate(validation_loader)):
        vinputs = vdata["images"]
        vinputs = vinputs.to(device)
        voutputs, vcommit_loss, predicate_matrix = model(vinputs)
        voutputs = voutputs.view(-1, NUM_FEATURES)
        out_count_SVHN.append(voutputs.sum(dim=1))
        running_out_attributes += voutputs.sum() / voutputs.shape[0]

avg_oa = running_out_attributes / (i + 1)
print(f"OA: {avg_oa}")

# =============================================================================================

import matplotlib.pyplot as plt

out_count_Cars = torch.cat(out_count_Cars).cpu()
out_count_AwA2 = torch.cat(out_count_AwA2).cpu()
out_count_SVHN = torch.cat(out_count_SVHN).cpu()

import numpy as np
plt.hist(out_count_Cars, bins=np.arange(min(out_count_Cars), max(out_count_Cars)+1), label="Cars Dist")
plt.hist(out_count_AwA2, bins=np.arange(min(out_count_AwA2), max(out_count_AwA2)+1), label="AwA2 Dist")
plt.hist(out_count_SVHN, bins=np.arange(min(out_count_SVHN), max(out_count_SVHN)+1), label="SVHN Dist")

plt.legend(loc='upper right')

plt.savefig('src/train/Cars/CarsOODAwA2NullSVHN.png')