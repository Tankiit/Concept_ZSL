import torchvision.transforms.v2 as transforms
import torch
import sys

sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/train/CUB")
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

NUM_EXCLUDE = 50

sys.path.insert(0, "/".join(__file__.split("/")[:-2]))
from util import get_CUB_test_labels
def make_ZSL_sets(train_transform, val_transform):
    indices = list(range(1, 201))
    
    test_labels = get_CUB_test_labels("src/ZSL/splits/CUB/CUBtestclasses.txt")
    val_labels = []
    train_indices = [x for x in indices if x not in test_labels]
    
    print(f"Training data: {200-len(test_labels)-len(val_labels)}, Testing data: {len(test_labels)}, Validation data: {len(val_labels)}")
    
    train_set = Cub2011("/storage/CUB", transform=train_transform, exclude = test_labels + val_labels)    
    test_set = Cub2011("/storage/CUB", transform=val_transform, train=False, exclude = test_labels + val_labels)
    
    ZSL_train_set = Cub2011("/storage/CUB", transform=val_transform, exclude = train_indices)
    ZSL_test_set = Cub2011("/storage/CUB", transform=val_transform, train=False, exclude = train_indices)
    
    return train_set, test_set, ZSL_train_set, ZSL_test_set

trainset, valset, ZSL_trainset, ZSL_valset = make_ZSL_sets(train_transform, val_transform)

BATCH_SIZE = 64

validation_loader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

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
        
        outputs, commit_loss, predicate_matrix = model(inputs)
        loss = criterion(outputs, labels, predicate_matrix) + commit_loss
        
        # first forward-backward pass
        loss.backward()
        optimizer.first_step(zero_grad=True)

        outputs, commit_loss, predicate_matrix = model(inputs)
        loss = criterion(outputs, labels, predicate_matrix) + commit_loss
        
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

NUM_CLASSES = 200
NUM_FEATURES = 80
EPOCHS = 50

FT_WEIGHT = 1

accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES - NUM_EXCLUDE, top_k=1).to(device)

sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/models")
from DeiT3AutoPredicates import ResExtr

model = ResExtr(NUM_FEATURES, NUM_CLASSES - NUM_EXCLUDE, deit_type=1).to(device)

base_optimizer = torch.optim.Adam
optimizer = SAM(model.parameters(), base_optimizer, lr=3e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer.base_optimizer, 1e-4, epochs=EPOCHS, steps_per_epoch=len(training_loader))
#scheduler = None

sys.path.insert(0, "/".join(__file__.split("/")[:-3]))
from SubsetLoss import BSSLoss
criterion = BSSLoss(NUM_FEATURES, pre_quantized=True).to(device)

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
            voutputs, vcommit_loss, predicate_matrix = model(vinputs)
            vloss = criterion(voutputs, vlabels, predicate_matrix) + vcommit_loss
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

print(f"Seen ACC: {avg_acc}, FP: {avg_fp}, MA: {avg_ma}, OA: {avg_oa}, Val Loss: {avg_vloss}, Train Loss: {avg_loss}")
    
torch.save(model.state_dict(), "CUBOfficialSplitDeiT3MeanAttrL.pt")
    
print("===============================================================")
print(f"Started Training On {NUM_EXCLUDE} Excluded Classes")

attributes_per_class = avg_oa.item() - avg_fp.item() + avg_ma.item()

validation_loader = torch.utils.data.DataLoader(
        ZSL_valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        ZSL_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

accuracy = Accuracy(task="multiclass", num_classes=NUM_EXCLUDE, top_k=1).to(device)

predis = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)

model.eval()
with torch.no_grad():
    for i, vdata in enumerate(training_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs, vcommit_loss, predicate_matrix = model(vinputs)
        
        for i in range(len(voutputs)):
            predis[vlabels[i]] += voutputs[i]
            
K = int(attributes_per_class+1)
topk, indices = torch.topk(predis, K, dim=1)
      
new_predicate_matrix = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)
new_predicate_matrix.scatter_(1, indices, 1)
    
model.classes = NUM_EXCLUDE
model.predicate_matrix = torch.nn.Parameter(new_predicate_matrix)

results = [[0,0]] * NUM_EXCLUDE
with torch.no_grad():
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs, vcommit_loss, predicate_matrix = model(vinputs)
        voutputs = voutputs.view(-1, 1, NUM_FEATURES)
        ANDed = voutputs * predicate_matrix
        diff = ANDed - voutputs
        preds = diff.sum(dim=2)
        # Add to results[label] the number of correct predictions and the number of predictions
        for i in range(len(preds)):
            results[vlabels[i]][0] += torch.argmax(preds[i]) == vlabels[i]
            results[vlabels[i]][1] += 1

accuracy = 0
for i in range(NUM_EXCLUDE):
    accuracy += results[i][0] / results[i][1]

unseen_acc = accuracy / NUM_EXCLUDE

print(f"Unseen ACC: {unseen_acc}")

harmonic_mean = 2 * avg_acc * unseen_acc / (avg_acc + unseen_acc)

print(f"Harmonic Mean: {harmonic_mean}")