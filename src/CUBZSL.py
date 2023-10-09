import torchvision.transforms as transforms
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

NUM_CLASSES = 200
NUM_EXCLUDE = 5
NUM_FEATURES = 64
EPOCHS = 30

FT_WEIGHT = 0

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-1]) + "/models")
from ResnetAutoPredicates import ResExtr

model = ResExtr(NUM_FEATURES, NUM_CLASSES - NUM_EXCLUDE, resnet_type=18, pretrained=True).to(device)
model.load_state_dict(torch.load("CUBRes18AutoPred195C.pt"))

print("===============================================================")
print("Started Training On 5 Excluded Classes")

sys.path.insert(0, "/".join(__file__.split("/")[:-1]) + "/train")
from CUBLoader import Cub2011

trainset = Cub2011("/storage/CUB", transform=train_transform, exclude=list(range(1, 196)))

valset = Cub2011("/storage/CUB", train=False, transform=val_transform, exclude=list(range(1, 196)))

#trainset = Cub2011("/storage/CUB", transform=train_transform, exclude=[196, 197, 198, 199, 200])
#valset = Cub2011("/storage/CUB", train=False, transform=val_transform, exclude=[196, 197, 198, 199, 200])

validation_loader = torch.utils.data.DataLoader(
        valset, batch_size=150, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=150, shuffle=False, num_workers=4)

old_predicate_matrix = torch.where(model.predicate_matrix > 0.5, torch.tensor(1.), torch.tensor(0.))

accuracy = Accuracy(task="multiclass", num_classes=NUM_EXCLUDE, top_k=1).to(device)

predis = []

def add_to_predis(voutputs, start, end):
    outs = voutputs[start:end, ].sum(dim=0)
    indices = torch.nonzero(outs > 11)
    features = torch.zeros((1, NUM_FEATURES))
    features[0, indices] = 1
    predis.append(features)

model.eval()
with torch.no_grad():
    for i, vdata in enumerate(training_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs, vcommit_loss, predicate_matrix = model(vinputs)
        
        add_to_predis(voutputs, 0, 29)
        
        add_to_predis(voutputs, 29, 59)
        
        add_to_predis(voutputs, 59, 89)
        
        add_to_predis(voutputs, 89, 119)
        
        add_to_predis(voutputs, 119, 149)
        
model.classes = 5
model.predicate_matrix = torch.nn.Parameter(torch.stack(predis).to(device))

running_vloss = 0.0
# Set the model to evaluation mode, disabling dropout and using population
# statistics for batch normalization.
running_acc = 0.0
running_false_positives = 0.0
running_missing_attr = 0.0
running_out_attributes = 0.0
with torch.no_grad():
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device) - 195
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