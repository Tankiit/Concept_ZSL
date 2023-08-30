import torch.nn as nn
import torch
from torchmetrics import Accuracy

def train_one_epoch():
    running_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data['features'].to(device), data['labels'].to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

if __name__ == "__main__":
    from numpy import loadtxt
    from datetime import datetime
    predicate_matrix_file = "datasets/Animals_with_Attributes2/predicate-matrix-binary.txt"
    predicate_matrix = torch.from_numpy(loadtxt(predicate_matrix_file, dtype=int)).cuda()

    NUM_FEATURES = 85
    NUM_CLASSES = 50

    torch.manual_seed(42)

    import pickle

    # pickle val_set
    with open("val_set.pkl", "rb") as f:
        val_set = pickle.load(f)

    # pickle train_set
    with open("train_set.pkl", "rb") as f:
        train_set = pickle.load(f)

    training_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 30

    model = nn.Sequential(
        nn.Linear(2048, 2048),
        nn.GELU(),
        nn.Linear(2048, NUM_CLASSES)
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    best_vloss = 1_000_000.

    best_loss = {
        "epoch": 0,
        "train_loss": 0,
        "val_loss": 0,
        "val_acc": 0,
    }

    best_acc = {
        "epoch": 0,
        "train_loss": 0,
        "val_loss": 0,
        "val_acc": 0,
    }

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch_number}")
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch()

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        running_acc = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata['features'].to(device), vdata['labels'].to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
                running_acc += accuracy(voutputs, vlabels)

        avg_vloss = running_vloss / (i + 1)
        avg_acc = running_acc / (i + 1)
        print(f"Training loss: {avg_loss}, Validation loss: {avg_vloss}, Validation accuracy: {avg_acc.item()}")

        epoch_number += 1
    
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_loss["epoch"] = epoch_number
            best_loss["train_loss"] = avg_loss
            best_loss["val_loss"] = avg_vloss
            best_loss["val_acc"] = avg_acc.item()

        if avg_acc > best_acc["val_acc"]:
            best_acc["epoch"] = epoch_number
            best_acc["train_loss"] = avg_loss
            best_acc["val_loss"] = avg_vloss
            best_acc["val_acc"] = avg_acc.item()

    print(f"Best loss: {best_loss}")
    print(f"Best accuracy: {best_acc}")