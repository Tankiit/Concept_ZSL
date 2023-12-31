import torch.nn as nn
import torch

torch.manual_seed(0)

import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/models")
from VectorAutoPredicates import ResExtr
from torchmetrics import Accuracy

def train_one_epoch():
    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        with torch.no_grad():
            inputs = resnet(inputs)

        outputs, commit_loss, predicate_matrix = model(inputs)

        # Compute the loss and its gradients
        if torch.any(torch.isnan(model.predicate_matrix)):
            exit()
        loss = loss_fn(outputs, labels, predicate_matrix) + commit_loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

eps=1e-10
def loss_fn(out, labels, predicate_matrix):
    out = out.view(-1, 1, NUM_FEATURES) # out is a batch of 1D binary vectors
    ANDed = out * predicate_matrix # AND operation
    diff = ANDed - out # Difference of ANDed and out => if equal, then out is a subset of its class' predicates

    entr_loss = nn.CrossEntropyLoss()
    loss_cl = entr_loss(diff.sum(dim=2), labels) # Is "out" a subset of its class' predicates?

    batch_size = out.shape[0]

    classes = torch.zeros(batch_size, NUM_CLASSES, device="cuda")
    classes[torch.arange(batch_size), labels] = 1
    classes = classes.view(batch_size, NUM_CLASSES, 1).expand(batch_size, NUM_CLASSES, NUM_FEATURES)

    extra_features = out - predicate_matrix + (out - predicate_matrix).pow(2)

    loss_neg_ft = torch.masked_select(extra_features, (1-classes).bool()).view(-1, NUM_FEATURES).sum() / batch_size

    labels_predicate = predicate_matrix[labels]
    extra_features_in = torch.masked_select(extra_features, classes.bool()).view(-1, NUM_FEATURES)
    loss_pos_ft = (labels_predicate - out.view(batch_size, NUM_FEATURES) + extra_features_in/2).sum() / batch_size

    return loss_cl + loss_neg_ft * FT_WEIGHT * loss_cl.item()/(loss_neg_ft.item() + eps) + loss_pos_ft * POS_FT_WEIGHT * loss_cl.item()/(loss_pos_ft.item() + eps)

if __name__ == "__main__":
    from datetime import datetime
    from torchvision import transforms, datasets

    # Data
    print('==> Preparing data..')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    validation_loader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    NUM_FEATURES = 32
    NUM_CLASSES = 10
    EPOCHS = 30
    accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

    POS_FT_WEIGHT = 0
    FT_WEIGHT = 0

    from torchvision.models import resnet50, ResNet50_Weights
    resnet = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
    resnet.fc = nn.Identity()
    for param in resnet.parameters():
        param.requires_grad = False
    model = ResExtr(2048, NUM_FEATURES, NUM_CLASSES).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_stats = {
        "epoch": 0,
        "train_loss": 0,
        "val_loss": 0,
        "val_acc": 0,
        "val_fp": 0,
    }

    from tqdm import tqdm
    for epoch in tqdm(range(EPOCHS)):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch()

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        running_acc = 0.0
        running_false_positives = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                vinputs = resnet(vinputs)
                voutputs, vcommit_loss, predicate_matrix = model(vinputs)
                vloss = loss_fn(voutputs, vlabels, predicate_matrix) + vcommit_loss
                running_vloss += vloss.item()
                voutputs = voutputs.view(-1, 1, NUM_FEATURES)
                ANDed = voutputs * predicate_matrix
                diff = ANDed - voutputs
                running_acc += accuracy(diff.sum(dim=2), vlabels)
                voutputs = voutputs.view(-1, NUM_FEATURES)
                running_false_positives += ((predicate_matrix[vlabels] - voutputs) == -1).sum() / voutputs.shape[0]

        avg_vloss = running_vloss / (i + 1)
        avg_acc = running_acc / (i + 1)
        avg_false_positives = running_false_positives / (i + 1)
        print(f"TRAIN_LOSS: {avg_loss}, VAL_LOSS: {avg_vloss}, ACC: {avg_acc}, FP: {avg_false_positives}")
        #print(model.bin_quantize._codebook.embed)

        if best_stats["val_acc"] < avg_acc:
            best_stats["epoch"] = epoch
            best_stats["train_loss"] = avg_loss
            best_stats["val_loss"] = avg_vloss
            best_stats["val_acc"] = avg_acc.item()
            best_stats["val_fp"] = avg_false_positives.item()

    print(best_stats)
