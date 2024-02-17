import torchvision.transforms as transforms
import torch

torch.set_float32_matmul_precision('medium')

train_transform =  transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(40),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/AwA2")
from AwA2Loader import AwA2Dataset
root = "datasets/Animals_with_Attributes2/JPEGImages"
awa2_dataset = AwA2Dataset(root=root, transform=val_transform)

all_indices = list(range(len(awa2_dataset)))
train_length = int(len(awa2_dataset) * 0.8)

import random
from torch.utils.data import Subset
train_idx = random.sample(all_indices, train_length)
val_idx = [i for i in all_indices if i not in train_idx]

zero_set = Subset(awa2_dataset, indices=train_idx)
val_set = Subset(awa2_dataset, indices=val_idx)

# Save the train and val sets
import pickle
with open("datasets/Animals_with_Attributes2/zero_set.pkl", "wb") as f:
    pickle.dump(zero_set, f)

with open("datasets/Animals_with_Attributes2/val_set.pkl", "wb") as f:
    pickle.dump(val_set, f)

import random
from CarsLoader import CarsZSLDataset
sys.path.insert(0, "/".join(__file__.split("/")[:-3]))
from LightningTrainerZeroOut import ZeroDataset, BSSTrainer
def make_sets(train_transform, val_transform):
    root_dir = "datasets/Stanford Cars"
    
    tr_dataset = CarsZSLDataset(root=root_dir, transform=train_transform)
    val_dataset = CarsZSLDataset(root=root_dir, transform=val_transform)

    all_indices = list(range(len(tr_dataset)))
    train_length = int(len(tr_dataset) * 0.8)

    train_idx = random.sample(all_indices, train_length)
    val_idx = [i for i in all_indices if i not in train_idx]

    train_img_set = Subset(tr_dataset, indices=train_idx)
    val_set = Subset(val_dataset, indices=val_idx)

    train_set = ZeroDataset(train_img_set, zero_set)

    return train_set, val_set

trainset, valset = make_sets(train_transform, val_transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

NUM_CLASSES = 196
NUM_FEATURES = 80
BATCH_SIZE = 64
EPOCHS = 30

import timm
modl = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True)
modl.head = torch.nn.Linear(512, NUM_FEATURES)

validation_loader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

lightning_model = BSSTrainer(modl, NUM_CLASSES, NUM_FEATURES, EPOCHS, training_loader, validation_loader)

import lightning as L
trainer = L.Trainer(devices=1, max_epochs=EPOCHS, precision=16)
trainer.fit(lightning_model)

# Save the model
from SubsetLoss import pack_model
packed = pack_model(lightning_model.model, lightning_model.criterion)
torch.save(packed.state_dict(), "models/Cars/DeiT3-80f-AwA2ZeroOut.pth")
