import torchvision.transforms as transforms
import torch

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-3]))
from LightningTrainer import BSSTrainer
lightning_model = BSSTrainer.load_from_checkpoint("models/Cars/DeiT3-80f.pth")

from SubsetLoss import pack_model
model = pack_model(lightning_model.model, lightning_model.criterion).to(device)

model.eval()

print("Model loaded")

root_dir = "datasets/Stanford Cars"

from CarsLoader import CarsZSLDataset
    
val_dataset = CarsZSLDataset(root=root_dir, transform=val_transform)
validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4)

running_out_attributes = 0.0

out_count_Cars = []

from tqdm import tqdm
with torch.no_grad():
    for i, vdata in tqdm(enumerate(validation_loader)):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs = model.get_features(vinputs)
        out_count_Cars.append(voutputs.sum(dim=1))
        running_out_attributes += voutputs.sum() / voutputs.shape[0]
        
avg_oa = running_out_attributes / (i + 1)
print(f"OA: {avg_oa}")

# =============================================================================================

sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/AwA2")
from AwA2Loader import AwA2Dataset
root = "datasets/Animals_with_Attributes2/JPEGImages"
val_dataset = AwA2Dataset(root=root, transform=val_transform)

validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4)

running_out_attributes = 0.0

out_count_AwA2 = []

with torch.no_grad():
    for i, vdata in tqdm(enumerate(validation_loader)):
        vinputs = vdata["images"]
        vinputs = vinputs.to(device)
        voutputs = model.get_features(vinputs)
        out_count_AwA2.append(voutputs.sum(dim=1))
        running_out_attributes += voutputs.sum() / voutputs.shape[0]

avg_oa = running_out_attributes / (i + 1)
print(f"OA: {avg_oa}")

# =============================================================================================

sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/SVHN")
from SVHNLoader import SVHNDataset
val_dataset = SVHNDataset(root='datasets/SVHN/train', transform=val_transform)

validation_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False, num_workers=4)

running_out_attributes = 0.0

out_count_SVHN = []

with torch.no_grad():
    for i, vdata in tqdm(enumerate(validation_loader)):
        vinputs = vdata["images"]
        vinputs = vinputs.to(device)
        voutputs = model.get_features(vinputs)
        out_count_SVHN.append(voutputs.sum(dim=1))
        running_out_attributes += voutputs.sum() / voutputs.shape[0]

avg_oa = running_out_attributes / (i + 1)
print(f"OA: {avg_oa}")

# =============================================================================================

import matplotlib.pyplot as plt

out_count_Cars = torch.cat(out_count_Cars).cpu()
out_count_AwA2 = torch.cat(out_count_AwA2).cpu()
out_count_SVHN = torch.cat(out_count_SVHN).cpu()

print(f"Cars std: {out_count_Cars.std()}")
print(f"AwA2 std: {out_count_AwA2.std()}")
print(f"SVHN std: {out_count_SVHN.std()}")

import numpy as np

plt.hist(out_count_Cars, weights=np.ones(len(out_count_Cars)) / len(out_count_Cars), bins=np.arange(min(out_count_Cars), max(out_count_Cars)+1), label="Cars")
plt.hist(out_count_AwA2, weights=np.ones(len(out_count_AwA2)) / len(out_count_AwA2), bins=np.arange(min(out_count_AwA2), max(out_count_AwA2)+1), label="AwA2")
plt.hist(out_count_SVHN, weights=np.ones(len(out_count_SVHN)) / len(out_count_SVHN), bins=np.arange(min(out_count_SVHN), max(out_count_SVHN)+1), label="SVHN")

from matplotlib.ticker import PercentFormatter
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

plt.xlabel("Number of Out Attributes")
plt.ylabel("Percentage of Samples")
plt.legend(loc='upper right')

#plt.savefig('src/train/Cars/CarsOODAwA2SVHN.png')

plt.show()