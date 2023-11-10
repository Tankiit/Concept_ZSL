from AwA2Loader import AwA2Dataset
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torch

train_transform =  transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
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

tr_dataset = AwA2Dataset(root='/storage/Animals_with_Attributes2/JPEGImages', transform=train_transform)
val_dataset = AwA2Dataset(root='/storage/Animals_with_Attributes2/JPEGImages', transform=val_transform)

all_indices = list(range(len(tr_dataset)))
train_length = int(len(tr_dataset) * 0.8)

import random
train_idx = random.sample(all_indices, train_length)
val_idx = [i for i in all_indices if i not in train_idx]

train_set = Subset(tr_dataset, indices=train_idx)
val_set = Subset(val_dataset, indices=val_idx)

print(f"Train set of length {len(train_set)} and val set of length {len(val_set)}")

validation_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4)

def train_one_epoch(scheduler):
    running_loss = 0.
    running_acc = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data["images"], data["labels"]
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        running_acc += accuracy(outputs, labels)
        loss = entr_loss(outputs, labels)
        
        # first forward-backward pass
        loss.backward()
        optimizer.first_step(zero_grad=True)

        outputs = model(inputs)
        loss = entr_loss(outputs, labels)
        
        # second forward-backward pass
        loss.backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)
        
        if scheduler is not None:
            scheduler.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1), running_acc / (i+1)

from torchmetrics import Accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

NUM_CLASSES = 50
EPOCHS = 10
accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

import timm
model = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True).to(device)
model.head = torch.nn.Linear(512, NUM_CLASSES).to(device)

import sys, os
sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/ZSL")
from sam import SAM

entr_loss = torch.nn.CrossEntropyLoss()

base_optimizer = torch.optim.Adam
optimizer = SAM(model.parameters(), base_optimizer, lr=3e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer.base_optimizer, 1e-4, epochs=EPOCHS, steps_per_epoch=len(training_loader))
#scheduler = None

best_stats = {
    "epoch": 0,
    "train_loss": 0,
    "val_loss": 0,
    "val_acc": 0,
    "train_acc": 0
}

from tqdm import tqdm
for epoch in tqdm(range(EPOCHS)):
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, avg_acc = train_one_epoch(scheduler)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    running_acc = 0.0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata["images"], vdata["labels"]
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = entr_loss(voutputs, vlabels)
            running_vloss += vloss.item()
            running_acc += accuracy(voutputs, vlabels)

    avg_vloss = running_vloss / (i + 1)
    avg_vacc = running_acc / (i + 1)
    print(f"TRAIN LOSS : {avg_loss}, VAL LOSS: {avg_vloss}, TRAIN ACC: {avg_acc}, VAL ACC: {avg_vacc}")

    if best_stats["val_acc"] < avg_acc:
        best_stats["epoch"] = epoch
        best_stats["train_loss"] = avg_loss
        best_stats["val_loss"] = avg_vloss
        best_stats["val_acc"] = avg_vacc.item()
        best_stats["train_acc"] = avg_acc.item()

print(best_stats)

