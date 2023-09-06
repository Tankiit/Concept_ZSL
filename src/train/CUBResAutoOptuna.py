import torch.nn as nn
import torch
import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/models")
from ResnetAutoPredicates import ResExtr
from torchmetrics import Accuracy
from CUBLoader import Cub2011

def train_one_epoch(model, optimizer, NUM_FEATURES, FT_WEIGHT, POS_FT_WEIGHT):
    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data["images"], data["labels"]
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs, commit_loss, predicate_matrix = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels, predicate_matrix, NUM_FEATURES, FT_WEIGHT, POS_FT_WEIGHT) + commit_loss
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

eps=1e-10
def loss_fn(out, labels, predicate_matrix, NUM_FEATURES, FT_WEIGHT, POS_FT_WEIGHT):
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import optim
from tqdm import tqdm
def objective(trial):
    global trial_num
    trial_num += 1
    print(f"Starting trial {trial_num}")
    NUM_FEATURES = trial.suggest_int("num_features", 1, 24)
    FT_WEIGHT = trial.suggest_float("ft_weight", 0, 3)
    #POS_FT_WEIGHT = trial.suggest_float("pos_ft_weight", 0, 2)
    POS_FT_WEIGHT = 0
    # Generate the model.
    model = ResExtr(NUM_FEATURES*16, NUM_CLASSES, resnet_type=18, pretrained=True).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 2e-5, 2e-3, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    EPOCHS = 30

    best_acc = 0.0

    for epoch in tqdm(range(EPOCHS)):
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        _ = train_one_epoch(model, optimizer, NUM_FEATURES*16, FT_WEIGHT, POS_FT_WEIGHT)

        model.eval()
        running_acc = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata["images"], vdata["labels"]
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs, _, predicate_matrix = model(vinputs)
                voutputs = voutputs.view(-1, 1, NUM_FEATURES*16)
                ANDed = voutputs * predicate_matrix
                diff = ANDed - voutputs
                running_acc += accuracy(diff.sum(dim=2), vlabels)

        avg_acc = running_acc / (i + 1)

        if avg_acc > best_acc:
            best_acc = avg_acc

        if epoch == 1 and best_acc < 0.1:
            raise optuna.TrialPruned()
        elif epoch == 4 and best_acc < 0.3:
            raise optuna.TrialPruned()
    
    return best_acc

if __name__ == "__main__":
    NUM_CLASSES = 200
    import optuna
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    # Data
    print('==> Preparing data..')
    train_transform =  transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    trainset = Cub2011("/storage/CUB", transform=train_transform)

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    ])

    valset = Cub2011("/storage/CUB", train=False, transform=val_transform)

    validation_loader = torch.utils.data.DataLoader(
            valset, batch_size=128, shuffle=False, num_workers=4)
    training_loader = torch.utils.data.DataLoader(
            trainset, batch_size=128, shuffle=True, num_workers=4)
    
    accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

    trial_num = -1

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        if key == "num_features":
            print("    {}: {}".format(key, value*16))
        else:
            print("    {}: {}".format(key, value))
            
# Trial 61 finished with value: 0.6621842980384827 and parameters: {'num_features': 5, 'ft_weight': 0.07777505402045351, 'ft_pos_weight': 0.6307639727798623, 'lr': 0.0001637430453184093}.
# Trial 93 finished with value: 0.6725443601608276 and parameters: {'num_features': 6, 'ft_weight': 0.11113670076359503, 'ft_pos_weight': 0.5169318900688755, 'lr': 0.00018327854921225235}.