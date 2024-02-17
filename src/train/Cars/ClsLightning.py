import torchvision.transforms.v2 as transforms
from torch.utils.data import Subset
import torch
import sys

torch.set_float32_matmul_precision('medium')

from CarsLoader import CarsZSLDataset

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

sys.path.insert(0, "/".join(__file__.split("/")[:-2]))
import random
def make_sets(train_transform, val_transform):
    root_dir = "datasets/Stanford Cars"
    
    tr_dataset = CarsZSLDataset(root=root_dir, transform=train_transform)
    val_dataset = CarsZSLDataset(root=root_dir, transform=val_transform)

    all_indices = list(range(len(tr_dataset)))
    train_length = int(len(tr_dataset) * 0.8)

    train_idx = random.sample(all_indices, train_length)
    val_idx = [i for i in all_indices if i not in train_idx]

    train_set = Subset(tr_dataset, indices=train_idx)
    val_set = Subset(val_dataset, indices=val_idx)

    return train_set, val_set

trainset, valset = make_sets(train_transform, val_transform)

BATCH_SIZE = 64
NUM_CLASSES = 196
NUM_FEATURES = 80
EPOCHS = 30

validation_loader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

import timm
model = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True)
model.head = torch.nn.Linear(512, NUM_FEATURES)

sys.path.insert(0, "/".join(__file__.split("/")[:-3]))
from LightningTrainer import BSSTrainer
lightning_model = BSSTrainer(model, NUM_CLASSES, NUM_FEATURES, EPOCHS, training_loader, validation_loader)

import lightning as L
trainer = L.Trainer(devices=1, max_epochs=EPOCHS, precision=16)
trainer.fit(lightning_model)

import os
if not os.path.exists("models/Cars"):
    os.makedirs("models/Cars")

# Save the model
trainer.save_checkpoint("models/Cars/DeiT3-80f.pth")