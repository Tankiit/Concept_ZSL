import torch
torch.set_float32_matmul_precision('medium')

import torchvision.transforms as transforms
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

NUM_CLASSES = 200
NUM_FEATURES = 80
EPOCHS = 50
NUM_EXCLUDE = 50
BATCH_SIZE = 64

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/train/CUB")
from CUBLoader import make_ZSL_sets
trainset, valset, ZSL_trainset, ZSL_valset = make_ZSL_sets("datasets/", NUM_EXCLUDE, train_transform, val_transform)
    
training_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
validation_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

import timm
model = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True)
model.head = torch.nn.Linear(512, NUM_FEATURES)

sys.path.insert(0, "/".join(__file__.split("/")[:-3]))
from LightningTrainerNoSAM import BSSTrainer
lightning_model = BSSTrainer(model, NUM_CLASSES, NUM_FEATURES, EPOCHS, training_loader, validation_loader, NUM_EXCLUDE=NUM_EXCLUDE)

import lightning as L
trainer = L.Trainer(devices=1, max_epochs=EPOCHS, precision=16)
trainer.fit(lightning_model)

print("===============================================================")
print(f"Started On {NUM_EXCLUDE} Excluded Classes")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Pack model
sys.path.insert(0, "/".join(__file__.split("/")[:-3]))
from SubsetLoss import pack_model
model = pack_model(lightning_model.model, lightning_model.criterion).to(device)
model.eval()

avg_oa = lightning_model.running_out_attributes / lightning_model.step_count
avg_fp = lightning_model.running_false_positives / lightning_model.step_count
avg_ma = lightning_model.running_missing_attr / lightning_model.step_count
attributes_per_class = avg_oa - avg_fp + avg_ma

ZSL_test_loader = torch.utils.data.DataLoader(
        ZSL_valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
ZSL_training_loader = torch.utils.data.DataLoader(
        ZSL_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

from torchmetrics import Accuracy
accuracy = Accuracy(task="multiclass", num_classes=NUM_EXCLUDE, top_k=1).to(device)

predis = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)

with torch.no_grad():
    for i, vdata in enumerate(ZSL_training_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        voutputs = model.model(vinputs)

        for i in range(len(voutputs)):
            predis[vlabels[i]] += voutputs[i]
    
    K = int(attributes_per_class+1)
    print(f"K: {K}")
    topk, indices = torch.topk(predis, K, dim=1)

    new_predicate_matrix = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)
    new_predicate_matrix.scatter_(1, indices, 1)
    
    catted = torch.cat((model.predicate_matrix, new_predicate_matrix), dim=0)

    model.predicate_matrix = torch.nn.Parameter(catted)

    unseen_results = [[0,0]] * NUM_EXCLUDE
    for i, vdata in enumerate(ZSL_test_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        preds = model(vinputs)
        # Add to results[label] the number of correct predictions and the number of predictions
        for i in range(len(preds)):
            unseen_results[vlabels[i]][0] += torch.argmax(preds[i]) == vlabels[i]+150
            unseen_results[vlabels[i]][1] += 1
            
    seen_results = [[0,0]] * (NUM_CLASSES-NUM_EXCLUDE)
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device)
        preds = model(vinputs)
        for i in range(len(preds)):
            seen_results[vlabels[i]][0] += torch.argmax(preds[i]) == vlabels[i]
            seen_results[vlabels[i]][1] += 1

unseen_acc = 0
for i in range(NUM_EXCLUDE):
    unseen_acc += unseen_results[i][0] / unseen_results[i][1]
    
seen_acc = 0
for i in range(NUM_CLASSES-NUM_EXCLUDE):
    seen_acc += seen_results[i][0] / seen_results[i][1]
    
unseen_acc = unseen_acc / NUM_EXCLUDE
seen_acc = seen_acc / (NUM_CLASSES-NUM_EXCLUDE)

print(f"Unseen ACC: {unseen_acc}, Seen ACC: {seen_acc}, Harmonic ACC: {2*seen_acc*unseen_acc/(seen_acc+unseen_acc)}")