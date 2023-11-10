import torchvision.transforms.v2 as transforms
import torch
import sys

sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/train")
from CUBLoader import Cub2011

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

import numpy as np
attributes_file = "/storage/CUB/CUB_200_2011/attributes/class_attribute_labels_continuous.txt"
base_predicate_matrix = np.loadtxt(attributes_file)
base_predicate_matrix[base_predicate_matrix > 0] = 1
base_predicate_matrix = torch.from_numpy(base_predicate_matrix).to(device)
print(f"Loaded predicate matrix of shape {base_predicate_matrix.shape}")

from util import get_CUB_test_labels
indices = list(range(1, 201))
test_labels = get_CUB_test_labels("src/ZSL/splits/CUB/CUBtestclasses.txt")
val_labels = []
train_indices = [x for x in indices if x not in test_labels]
print(f"Training data: {200-len(test_labels)-len(val_labels)}, Testing data: {len(test_labels)}, Validation data: {len(val_labels)}")

predicate_matrix = base_predicate_matrix[[i-1 for i in train_indices]]
print(f"Training with predicate matrix of shape {predicate_matrix.shape}")

new_predicate_matrix = base_predicate_matrix[[i-1 for i in test_labels]]
print(f"Testing with predicate matrix of shape {new_predicate_matrix.shape}")

def make_ZSL_sets(train_transform, val_transform):
    train_set = Cub2011("/storage/CUB", transform=train_transform, exclude = test_labels + val_labels)    
    test_set = Cub2011("/storage/CUB", transform=val_transform, train=False, exclude = test_labels + val_labels)
    
    ZSL_set = Cub2011("/storage/CUB", transform=val_transform, all_data=True, exclude = train_indices)
    
    return train_set, test_set, ZSL_set

trainset, valset, ZSL_set = make_ZSL_sets(train_transform, val_transform)

BATCH_SIZE = 64

validation_loader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

eps=1e-10
def loss_fn(out, labels):
    out = out.view(-1, 1, NUM_FEATURES) # out is a batch of 1D binary vectors
    entr_loss = torch.nn.CrossEntropyLoss()
    loss_cl = entr_loss(-(predicate_matrix - out).pow(2).sum(dim=2), labels) # Is "out" a subset of its class' predicates?
    
    batch_size = out.shape[0]

    out = out.view(-1, NUM_FEATURES)
    diff_square = (out - predicate_matrix[labels]).pow(2)
    
    false_positives = (out - predicate_matrix[labels] + diff_square).sum() / batch_size
    missing_attr = (predicate_matrix[labels] - out + diff_square).sum() / batch_size
    
    loss_ft = (1 + false_positives + missing_attr)
    
    loss_ft *= loss_cl.item()/(loss_ft.item() + eps)
    
    return loss_cl + loss_ft * FT_WEIGHT

from sam import SAM
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
        
        outputs, commit_loss = model(inputs)
        loss = loss_fn(outputs, labels) + commit_loss
        
        # first forward-backward pass
        loss.backward()
        optimizer.first_step(zero_grad=True)

        outputs, commit_loss = model(inputs)
        loss = loss_fn(outputs, labels) + commit_loss
        
        # second forward-backward pass
        loss.backward()  # make sure to do a full forward pass
        optimizer.second_step(zero_grad=True)
        
        if scheduler is not None:
            scheduler.step()
    
        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

from torchmetrics import Accuracy

NUM_CLASSES = 200
NUM_EXCLUDE = 50
NUM_FEATURES = predicate_matrix.shape[1]
EPOCHS = 50

FT_WEIGHT = 1

accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES - NUM_EXCLUDE, top_k=1).to(device)

sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/models")
from DeiT3NoAuto import ResExtr

model = ResExtr(NUM_FEATURES, NUM_CLASSES - NUM_EXCLUDE, deit_type=1).to(device)

base_optimizer = torch.optim.Adam
optimizer = SAM(model.parameters(), base_optimizer, lr=3e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer.base_optimizer, 1e-4, epochs=EPOCHS, steps_per_epoch=len(training_loader))
#scheduler = None

from tqdm import tqdm
for epoch in tqdm(range(EPOCHS)):
    model.train(True)
    avg_loss = train_one_epoch(scheduler)

    running_vloss = 0.0

    model.eval()
    running_acc = 0.0
    running_false_positives = 0.0
    running_missing_attr = 0.0
    running_out_attributes = 0.0

    with torch.no_grad():
        for i, vdata in enumerate(validation_loader):
            vinputs, vlabels = vdata["images"], vdata["labels"]
            vinputs = vinputs.to(device)
            vlabels = vlabels.to(device)
            voutputs, vcommit_loss = model(vinputs)
            vloss = loss_fn(voutputs, vlabels) + vcommit_loss
            running_vloss += vloss.item()
            voutputs = voutputs.view(-1, 1, NUM_FEATURES)
            running_acc += accuracy(-(predicate_matrix - voutputs).pow(2).sum(dim=2), vlabels)
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

print(f"Seen ACC: {avg_acc}, FP: {avg_fp}, MA: {avg_ma}, OA: {avg_oa}, Val Loss: {avg_vloss}, Train Loss: {avg_loss}")
    
torch.save(model.state_dict(), "CUBOfficialSplitDeiT3NoAuto.pt")
    
print("===============================================================")
print(f"Started Training On {NUM_EXCLUDE} Excluded Classes")

attributes_per_class = avg_oa.item() - avg_fp.item() + avg_ma.item()

zsl_loader = torch.utils.data.DataLoader(
        ZSL_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
results = [[0,0]] * NUM_EXCLUDE
with torch.no_grad():
    for i, vdata in enumerate(zsl_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs, vcommit_loss = model(vinputs)
        voutputs = voutputs.view(-1, 1, NUM_FEATURES)
        preds = -(predicate_matrix - voutputs).pow(2).sum(dim=2)
        # Add to results[label] the number of correct predictions and the number of predictions
        for i in range(len(preds)):
            results[vlabels[i]][0] += torch.argmax(preds[i]) == vlabels[i]
            results[vlabels[i]][1] += 1

accuracy = 0
for i in range(NUM_EXCLUDE):
    accuracy += results[i][0] / results[i][1]

unseen_acc = accuracy / NUM_EXCLUDE

print(f"Unseen ACC: {unseen_acc}")
