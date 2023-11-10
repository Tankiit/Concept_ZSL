import torchvision.transforms.v2 as transforms
from torch.utils.data import Subset
import torch
import sys

sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/train/Cars")
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
def make_ZSL_sets(train_transform, val_transform):
    path = "src/ZSL/splits/Cars/classes.txt"

    with open(path, "r") as f:
        classes = f.readlines()
        classes = [x.strip() for x in classes]
        
    random.shuffle(classes)
    test_labels = classes[:NUM_EXCLUDE]
    train_labels = classes[NUM_EXCLUDE:]
    
    val_labels = []
    
    print(f"Training data: {196-len(test_labels)-len(val_labels)}, Testing data: {len(test_labels)}, Validation data: {len(val_labels)}")
    
    root_dir = "/storage/Cars"
    
    tr_dataset = CarsZSLDataset(root=root_dir, transform=train_transform, exclude=test_labels)
    val_dataset = CarsZSLDataset(root=root_dir, transform=val_transform, exclude=test_labels)

    all_indices = list(range(len(tr_dataset)))
    train_length = int(len(tr_dataset) * 0.8)

    train_idx = random.sample(all_indices, train_length)
    val_idx = [i for i in all_indices if i not in train_idx]

    train_set = Subset(tr_dataset, indices=train_idx)
    val_set = Subset(val_dataset, indices=val_idx)
    
    IMAGES_PER_CLASS = 10
    
    ZSL_test_set = CarsZSLDataset(root=root_dir, transform=val_transform, exclude=train_labels, skip_first_n=IMAGES_PER_CLASS)
    ZSL_train_set = CarsZSLDataset(root=root_dir, transform=val_transform, exclude=train_labels, end_at_n=IMAGES_PER_CLASS)

    return train_set, val_set, ZSL_train_set, ZSL_test_set

NUM_EXCLUDE = 49

trainset, valset, ZSL_train_set, ZSL_test_set = make_ZSL_sets(train_transform, val_transform)

BATCH_SIZE = 64

validation_loader = torch.utils.data.DataLoader(
        valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

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
    
    loss_mean_attr = (predicate_matrix.sum(dim=1).mean() - NUM_FEATURES//2).pow(2)
    loss_ft = 2*loss_mean_attr + false_positives + missing_attr
    loss_ft *= loss_cl.item()/(loss_ft.item() + eps)
    
    return loss_cl + loss_ft * FT_WEIGHT

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

from torchmetrics import Accuracy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

NUM_CLASSES = 196
NUM_FEATURES = 80
EPOCHS = 30

FT_WEIGHT = 0.7

accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES - NUM_EXCLUDE, top_k=1).to(device)

sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/models")
from DeiT3AutoPredicates import ResExtr

model = ResExtr(NUM_FEATURES, NUM_CLASSES - NUM_EXCLUDE).to(device)

from sam import SAM
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

print(f"Seen ACC: {avg_acc}, FP: {avg_fp}, MA: {avg_ma}, OA: {avg_oa}, Val Loss: {avg_vloss}, Train Loss: {avg_loss}")
    
print("===============================================================")
print(f"Started Training On {NUM_EXCLUDE} Excluded Classes")

attributes_per_class = avg_oa.item() - avg_fp.item() + avg_ma.item()

ZSL_training_loader = torch.utils.data.DataLoader(
        ZSL_train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
ZSL_test_loader = torch.utils.data.DataLoader(
        ZSL_test_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

accuracy = Accuracy(task="multiclass", num_classes=NUM_EXCLUDE, top_k=1).to(device)

predis = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)

model.eval()
with torch.no_grad():
    for i, vdata in enumerate(ZSL_training_loader):
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
    
results = [[0,0]] * NUM_EXCLUDE
with torch.no_grad():
    for i, vdata in enumerate(ZSL_test_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs, _, _ = model(vinputs)
        voutputs = voutputs.view(-1, 1, NUM_FEATURES)
        ANDed = voutputs * new_predicate_matrix
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

print(f"Unseen ACC avg outputs: {unseen_acc}")

K = NUM_FEATURES//2
topk, indices = torch.topk(predis, K, dim=1)
      
new_predicate_matrix = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)
new_predicate_matrix.scatter_(1, indices, 1)
    
results = [[0,0]] * NUM_EXCLUDE
with torch.no_grad():
    for i, vdata in enumerate(ZSL_test_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs, _, _ = model(vinputs)
        voutputs = voutputs.view(-1, 1, NUM_FEATURES)
        ANDed = voutputs * new_predicate_matrix
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

print(f"Unseen ACC medium outputs: {unseen_acc}")