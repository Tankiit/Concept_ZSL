import torch, sys
torch.set_float32_matmul_precision('medium')

from torch.utils.data import Subset

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

sys.path.insert(0, "/".join(__file__.split("/")[:-2]))
import random
sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/train/Cars")
from CarsLoader import CarsZSLDataset
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
    
    root_dir = "datasets/Stanford Cars"
    
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
NUM_CLASSES = 196
NUM_FEATURES = 80
EPOCHS = 30
BATCH_SIZE = 64

trainset, valset, ZSL_trainset, ZSL_valset = make_ZSL_sets(train_transform, val_transform)

training_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
validation_loader = torch.utils.data.DataLoader(valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

import timm
model = timm.create_model("deit3_medium_patch16_224.fb_in22k_ft_in1k", pretrained=True)
model.head = torch.nn.Linear(512, NUM_FEATURES)

sys.path.insert(0, "/".join(__file__.split("/")[:-3]))
from LightningTrainer import BSSTrainer
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
attributes_per_class = int(avg_oa - avg_fp + avg_ma) + 1

unseen_training_loader = torch.utils.data.DataLoader(
        ZSL_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
unseen_val_loader = torch.utils.data.DataLoader(
        ZSL_valset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

predis = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)

with torch.no_grad():
    for i, vdata in enumerate(unseen_training_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device) - 1
        voutputs = model.get_features(vinputs)

        for j in range(len(voutputs)):
            predis[vlabels[j]] += voutputs[j]

    topk, indices = torch.topk(predis, attributes_per_class, dim=1)

    new_predicate_matrix = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)
    new_predicate_matrix.scatter_(1, indices, 1)

    catted = torch.cat((model.predicate_matrix, new_predicate_matrix), dim=0)
    model.predicate_matrix = torch.nn.Parameter(catted)

    unseen_results = [[0,0]] * NUM_EXCLUDE
    for i, vdata in enumerate(unseen_val_loader):
        vinputs, vlabels = vdata["images"], vdata["labels"]
        vinputs = vinputs.to(device)
        vlabels = vlabels.to(device) - 1
        preds = model(vinputs)
        for i in range(len(preds)):
            unseen_results[vlabels[i]][0] += torch.argmax(preds[i]) == vlabels[i]+NUM_CLASSES-NUM_EXCLUDE
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