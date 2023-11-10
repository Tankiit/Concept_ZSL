from CUBLoader import Cub2011
import torchvision.transforms as transforms
import torch

torch.manual_seed(42)

train_transform =  transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

trainset = Cub2011("/storage/CUB", transform=train_transform)

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

valset = Cub2011("/storage/CUB", train=False, transform=val_transform)

validation_loader = torch.utils.data.DataLoader(
        valset, batch_size=128, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4)

def train_one_epoch_baseline(scheduler):
    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data["images"], data["labels"]
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

from torchmetrics import Accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

NUM_CLASSES = 200
EPOCHS = 20
accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

from torchvision.models import vit_b_16
import torch.nn as nn
torch.backends.cudnn.benchmark = True
model = vit_b_16(weights="DEFAULT")

model.heads[-1] = nn.Linear(768, NUM_CLASSES)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-4, epochs=EPOCHS, steps_per_epoch=len(training_loader))
#scheduler = None

best_stats = {
    "epoch": 0,
    "train_loss": 0,
    "val_loss": 0,
    "val_acc": 0,
}

from tqdm import tqdm
for epoch in tqdm(range(EPOCHS)):
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch_baseline(scheduler)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    running_acc = 0.0
    running_false_positives = 0.0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata["images"], vdata["labels"]
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs = model(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss.item()
            running_acc += accuracy(voutputs, vlabels)

    avg_vloss = running_vloss / (i + 1)
    avg_acc = running_acc / (i + 1)
    print(f"LOSS: {avg_vloss}, ACC: {avg_acc}")
    with open("CUBRes18ClsOneCycle.csv", "a") as f:
        f.write(f"{epoch}, {avg_loss}, {avg_vloss}, {avg_acc}\n")

    if best_stats["val_acc"] < avg_acc:
        best_stats["epoch"] = epoch
        best_stats["train_loss"] = avg_loss
        best_stats["val_loss"] = avg_vloss
        best_stats["val_acc"] = avg_acc.item()

print(best_stats)