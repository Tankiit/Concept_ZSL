import torch.nn as nn
import torch
import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-2]) + "/models")
from ResnetAutoPredicates import ResExtr
from torchmetrics import Accuracy

def train_one_epoch(model, optimizer, NUM_FEATURES, FT_WEIGHT, POS_FT_WEIGHT):
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
        outputs, commit_loss, predicate_matrix = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels, predicate_matrix, NUM_FEATURES, FT_WEIGHT, POS_FT_WEIGHT) + commit_loss
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

eps = 1e-10
def loss_fn(out, labels, predicate_matrix, NUM_FEATURES, FT_WEIGHT, POS_FT_WEIGHT):
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from torch import optim
from tqdm import tqdm
def objective(trial):
    global trial_num
    trial_num += 1
    print(f"Starting trial {trial_num}")
    NUM_FEATURES = trial.suggest_int("num_features", 1, 4)
    FT_WEIGHT = trial.suggest_float("ft_weight", 0, 1.5)
    POS_FT_WEIGHT = trial.suggest_float("ft_pos_weight", 0, 1.5)
    # Generate the model.
    model = ResExtr(NUM_FEATURES*16, NUM_CLASSES).to(device)

    # Generate the optimizers.
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    EPOCHS = 15

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
                vinputs, vlabels = vdata
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

        if epoch == 1 and best_acc < 0.15:
            raise optuna.TrialPruned()
        elif epoch == 4 and best_acc < 0.5:
            raise optuna.TrialPruned()
    
    return best_acc

if __name__ == "__main__":
    NUM_CLASSES = 10
    import optuna
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    training_loader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    validation_loader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=4)
    
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