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

validation_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=4)

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

        outputs, commit_loss, predicate_matrix = model(inputs)
        loss = loss_fn(outputs, labels, predicate_matrix) + commit_loss
        
        # first forward-backward pass
        loss.backward()
        optimizer.first_step(zero_grad=True)

        outputs, commit_loss, predicate_matrix = model(inputs)
        loss = loss_fn(outputs, labels, predicate_matrix) + commit_loss
        
        # second forward-backward pass
        loss.backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)
        
        if scheduler is not None:
            scheduler.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

eps=1e-10
def loss_fn(out, labels, predicate_matrix):
    out = out.view(-1, 1, NUM_FEATURES) # out is a batch of 1D binary vectors
    ANDed = out * predicate_matrix # AND operation
    diff = ANDed - out # Difference of ANDed and out => if equal, then out is a subset of its class' predicates

    entr_loss = torch.nn.CrossEntropyLoss()
    loss_cl = entr_loss(diff.sum(dim=2), labels) # Is "out" a subset of its class' predicates?

    batch_size = out.shape[0]

    out = out.view(-1, NUM_FEATURES)
    diff_square = (out - predicate_matrix[labels]).pow(2)
    
    false_positives = (out - predicate_matrix[labels] + diff_square).sum() / batch_size
    missing_attr = (predicate_matrix[labels] - out + diff_square).sum() / batch_size
    
    loss_ft = (1 + false_positives + missing_attr)
    loss_ft *= loss_cl.item()/(loss_ft.item() + eps)
    
    return loss_cl + loss_ft * FT_WEIGHT

from torchmetrics import Accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

NUM_CLASSES = 50
NUM_FEATURES = 24
EPOCHS = 50
accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

FT_WEIGHT = 0.7

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/models")
from DeiT3AutoPredicates import ResExtr

model = ResExtr(NUM_FEATURES, NUM_CLASSES).to(device)

sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/ZSL")
from sam import SAM

base_optimizer = torch.optim.Adam
optimizer = SAM(model.parameters(), base_optimizer, lr=3e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer.base_optimizer, 1e-4, epochs=EPOCHS, steps_per_epoch=len(training_loader))
#scheduler = None

best_stats = {
    "epoch": 0,
    "train_loss": 0,
    "val_loss": 0,
    "val_acc": 0,
    "fp": 0,
    "ma": 0,
    "oa": 0
}

from tqdm import tqdm
for epoch in tqdm(range(EPOCHS)):
    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss = train_one_epoch(scheduler)

    running_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    running_acc = 0.0
    running_false_positives = 0.0
    running_missing_attr = 0.0
    running_out_attributes = 0.0

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata["images"], vdata["labels"]
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs, vcommit_loss, predicate_matrix = model(vinputs)
            vloss = loss_fn(voutputs, vlabels, predicate_matrix) + vcommit_loss
            running_vloss += vloss.item()
            voutputs = voutputs.view(-1, 1, NUM_FEATURES)
            ANDed = voutputs * predicate_matrix
            diff = ANDed - voutputs
            running_acc += accuracy(diff.sum(dim=2), vlabels)
            voutputs = voutputs.view(-1, NUM_FEATURES)
            running_false_positives += ((predicate_matrix[vlabels] - voutputs) == -1).sum() / voutputs.shape[0]
            running_missing_attr += ((voutputs - predicate_matrix[vlabels]) == -1).sum() / voutputs.shape[0]
            running_out_attributes += voutputs.sum() / voutputs.shape[0]

    avg_vloss = running_vloss / (i + 1)
    avg_acc = running_acc / (i + 1)
    avg_fp = running_false_positives / (i + 1)
    avg_ma = running_missing_attr / (i + 1)
    avg_oa = running_out_attributes / (i + 1)
    print(f"LOSS: {avg_vloss}, ACC: {avg_acc}, FP: {avg_fp}, MA: {avg_ma}, OA: {avg_oa}")
    #with open("CUBRes18AutoPredData.csv", "a") as f:
        #f.write(f"{epoch}, {avg_loss}, {avg_vloss}, {avg_acc}, {avg_fp}, {avg_ma}, {avg_oa}\n")

    if best_stats["val_acc"] < avg_acc:
        best_stats["epoch"] = epoch
        best_stats["train_loss"] = avg_loss
        best_stats["val_loss"] = avg_vloss
        best_stats["val_acc"] = avg_acc.item()
        best_stats["fp"] = avg_fp.item()
        best_stats["ma"] = avg_ma.item()
        best_stats["oa"] = avg_oa.item()

print(best_stats)

#torch.save(model.state_dict(), "CUBVGGAutoPred.pt")