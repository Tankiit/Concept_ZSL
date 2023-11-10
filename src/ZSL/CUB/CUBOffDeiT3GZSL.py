import torchvision.transforms.v2 as transforms
import torch
import sys

sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/train/CUB")
from CUBLoader import Cub2011, make_train_split_file

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
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

NUM_EXCLUDE = 50
BATCH_SIZE = 64
NUM_CLASSES = 200
NUM_FEATURES = 80

sys.path.insert(0, "/".join(__file__.split("/")[:-2]))
from util import get_CUB_test_labels
test_labels = get_CUB_test_labels("src/ZSL/splits/CUB/CUBtestclasses.txt")
test_set = Cub2011("/storage/CUB", transform=val_transform, train=False, exclude = test_labels)
train_set = Cub2011("/storage/CUB", transform=train_transform, exclude = test_labels)

seen_train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
seen_val_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

from torchmetrics import Accuracy
saccuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES - NUM_EXCLUDE, top_k=1).to(device)

sys.path.insert(0, "/".join(__file__.split("/")[:-3]) + "/models")
from DeiT3AutoPredicates import ResExtr

model = ResExtr(NUM_FEATURES, NUM_CLASSES - NUM_EXCLUDE, deit_type=1).to(device)
model.load_state_dict(torch.load("CUBOfficialSplitDeiT3MeanAttrL.pt"))
model.eval()

predis = torch.zeros(NUM_CLASSES - NUM_EXCLUDE, NUM_FEATURES).to(device)

attributes_per_class = 40

print("===============================================================")
def make_ZSL_sets(val_transform, split_file):
        indices = list(range(1, 201))
        train_indices = [x for x in indices if x not in test_labels]
        
        ZSL_train_set = Cub2011("/storage/CUB", transform=val_transform, train_split_file=split_file, exclude = train_indices)
        ZSL_test_set = Cub2011("/storage/CUB", transform=val_transform, train=False, train_split_file=split_file, exclude = train_indices)

        return ZSL_train_set, ZSL_test_set

print(f"Started Building On {NUM_EXCLUDE} Excluded Classes")

def get_acc_avg_n_img_per_class(image_count, iters, split_file):
    unseen_accuracies = []
    seen_accuracies = []
    for i in range(iters):
        if split_file is not None:
            make_train_split_file(image_count, split_file, "/storage/CUB")

        ZSL_trainset, ZSL_valset = make_ZSL_sets(val_transform, split_file)

        unseen_val_loader = torch.utils.data.DataLoader(ZSL_valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        unseen_training_loader = torch.utils.data.DataLoader(ZSL_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        predis = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)

        with torch.no_grad():
            for i, vdata in enumerate(unseen_training_loader):
                vinputs, vlabels = vdata["images"], vdata["labels"]
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs, _, predicate_matrix = model(vinputs)

                for i in range(len(voutputs)):
                    predis[vlabels[i]] += voutputs[i]
            
            topk, indices = torch.topk(predis, attributes_per_class, dim=1)

            new_predicate_matrix = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)
            new_predicate_matrix.scatter_(1, indices, 1)
            
            catted = torch.cat((predicate_matrix, new_predicate_matrix), dim=0)

            unseen_results = [[0,0]] * NUM_EXCLUDE
            for i, vdata in enumerate(unseen_val_loader):
                vinputs, vlabels = vdata["images"], vdata["labels"]
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs, _, _ = model(vinputs)
                voutputs = voutputs.view(-1, 1, NUM_FEATURES)
                ANDed = voutputs * catted
                diff = ANDed - voutputs
                preds = diff.sum(dim=2)
                # Add to results[label] the number of correct predictions and the number of predictions
                for i in range(len(preds)):
                    unseen_results[vlabels[i]][0] += torch.argmax(preds[i]) == vlabels[i]+150
                    unseen_results[vlabels[i]][1] += 1
                    
            seen_results = [[0,0]] * (NUM_CLASSES-NUM_EXCLUDE)
            for i, vdata in enumerate(seen_val_loader):
                vinputs, vlabels = vdata["images"], vdata["labels"]
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs, _, _ = model(vinputs)
                voutputs = voutputs.view(-1, 1, NUM_FEATURES)
                ANDed = voutputs * catted
                diff = ANDed - voutputs
                preds = diff.sum(dim=2)
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
        unseen_accuracies.append(unseen_acc.item())
        seen_accuracies.append(seen_acc.item())

        unseen_acc = sum(unseen_accuracies)/len(unseen_accuracies)
        seen_acc = sum(seen_accuracies)/len(seen_accuracies)
        print(f"Unseen ACC: {unseen_acc}, Seen ACC: {seen_acc}, Harmonic ACC: {2*seen_acc*unseen_acc/(seen_acc+unseen_acc)}")
        
    print(unseen_accuracies)
    print(max(unseen_accuracies), min(unseen_accuracies))

#split_file=None
split_file = "/notebooks/Concept_ZSL/src/ZSL/splits/CUB/lowsplits.txt"

get_acc_avg_n_img_per_class(10, 30, split_file)