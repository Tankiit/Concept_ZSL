import torchvision.transforms.v2 as transforms
from torch.utils.data import Subset
import torch
import sys

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
    root_dir = "/storage/Cars"
    
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

validation_loader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

def train_one_epoch(scheduler):
    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data["images"], data["labels"]
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        # first forward-backward pass
        loss.backward()
        optimizer.first_step(zero_grad=True)

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        
        # second forward-backward pass
        loss.backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)
        
        if scheduler is not None:
            scheduler.step()
    
        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

from torchmetrics import Accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

NUM_CLASSES = 196
EPOCHS = 10

accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

import timm
model = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True).to(device)
model.head = torch.nn.Linear(512, NUM_CLASSES).to(device)

sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/ZSL")
from sam import SAM
base_optimizer = torch.optim.Adam
optimizer = SAM(model.parameters(), base_optimizer, lr=3e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer.base_optimizer, 1e-4, epochs=EPOCHS, steps_per_epoch=len(training_loader))
#scheduler = None

loss_fn = torch.nn.CrossEntropyLoss()

best_stats = {
    "epoch": 0,
    "train_loss": 0,
    "val_loss": 0,
    "val_acc": 0,
}

from tqdm import tqdm
for epoch in tqdm(range(EPOCHS)):
    model.train(True)
    avg_loss = train_one_epoch(scheduler)

    running_vloss = 0.0

    model.eval()
    running_acc = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata["images"], vdata["labels"]
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss.item()
            running_acc += accuracy(voutputs, vlabels)

    avg_vloss = running_vloss / (i + 1)
    avg_acc = running_acc / (i + 1)
    
    if best_stats["val_acc"] < avg_acc:
        best_stats["epoch"] = epoch
        best_stats["train_loss"] = avg_loss
        best_stats["val_loss"] = avg_vloss
        best_stats["val_acc"] = avg_acc.item()
    
    print(f"LOSS: {avg_vloss}, ACC: {avg_acc}")

print(best_stats)