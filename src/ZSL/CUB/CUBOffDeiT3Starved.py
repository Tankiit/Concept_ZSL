import torchvision.transforms.v2 as transforms
import torch
import sys

sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/train")
from CUBLoader import Cub2011, make_train_split_file

train_transform =  transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

NUM_CLASSES = 200
NUM_FEATURES = 80

sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/models")
from DeiT3AutoPredicates import ResExtr

model = ResExtr(NUM_FEATURES, NUM_CLASSES - NUM_EXCLUDE, deit_type=1).to(device)
avg_acc = 0.887456476688385
model.load_state_dict(torch.load("CUBOfficialSplitDeiT3.pt"))

print("===============================================================")
def make_ZSL_sets(val_transform):
        indices = list(range(1, 201))

        test_labels = get_CUB_test_labels("src/ZSL/splits/CUB/CUBtestclasses.txt")
        train_indices = [x for x in indices if x not in test_labels]

        ZSL_train_set = Cub2011("/storage/CUB", transform=val_transform, train_split_file=split_file, exclude = train_indices)
        ZSL_test_set = Cub2011("/storage/CUB", transform=val_transform, train=False, train_split_file=split_file, exclude = train_indices)

        return ZSL_train_set, ZSL_test_set

print(f"Started Building On {NUM_EXCLUDE} Excluded Classes")

attributes_per_class = 25

BATCH_SIZE = 64

from util import get_CUB_test_labels
from tqdm import tqdm
def get_acc_avg_n_img_per_class(image_count, samples, split_file):
    accuracies = []
    for i in tqdm(range(samples)):
        make_train_split_file(image_count, split_file, "/storage/CUB")

        ZSL_trainset, ZSL_valset = make_ZSL_sets(val_transform)

        validation_loader = torch.utils.data.DataLoader(
                ZSL_valset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
        training_loader = torch.utils.data.DataLoader(
                ZSL_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

        predis = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)

        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(training_loader):
                vinputs, vlabels = vdata["images"], vdata["labels"]
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs, _, _ = model(vinputs)

                for i in range(len(voutputs)):
                    predis[vlabels[i]] += voutputs[i]

            K = int(attributes_per_class+1)
            topk, indices = torch.topk(predis, K, dim=1)

            new_predicate_matrix = torch.zeros(NUM_EXCLUDE, NUM_FEATURES).to(device)
            new_predicate_matrix.scatter_(1, indices, 1)

            results = [[0,0]] * NUM_EXCLUDE
            for i, vdata in enumerate(validation_loader):
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
        accuracies.append(unseen_acc.item())

        print(f"Unseen ACC: {sum(accuracies)/len(accuracies)}")
        
    print(accuracies)
    print(max(accuracies), min(accuracies))

#split_file=None
split_file = "/notebooks/Concept_ZSL/src/ZSL/splits/CUB/lowsplits.txt"

get_acc_avg_n_img_per_class(30, 100, split_file)