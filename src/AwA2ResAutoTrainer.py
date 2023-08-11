import torch.nn as nn
import torch
from ResnetAutoPredicates import ResExtr
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
        outputs, commit_loss, predicate_matrix = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels, predicate_matrix) + commit_loss
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

    return running_loss / (i+1)

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

        return loss_cl + loss_neg_ft * FT_WEIGHT * loss_cl.item()/loss_neg_ft.item() + loss_pos_ft * POS_FT_WEIGHT * loss_cl.item()/loss_pos_ft.item()

if __name__ == "__main__":
    from datetime import datetime

    NUM_FEATURES = 64
    NUM_CLASSES = 50
    FT_WEIGHT = 0.7

    POS_FT_WEIGHT = 0.0
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

    model = ResExtr(2048, NUM_FEATURES, 1, NUM_CLASSES).to(device)

    accuracy = Accuracy(task="multiclass", num_classes=NUM_CLASSES, top_k=1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=1e-5)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print(f"EPOCH {epoch}")
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
                vinputs, vlabels = vdata['features'].to(device), vdata['labels'].to(device)
                voutputs, vcommit_loss, predicate_matrix = model(vinputs)
                vloss = loss_fn(voutputs, vlabels, predicate_matrix) + vcommit_loss
                running_vloss += vloss.item()
                voutputs = voutputs.view(-1, 1, NUM_FEATURES)
                ANDed = voutputs * predicate_matrix
                diff = ANDed - voutputs
                running_acc += accuracy(diff.sum(dim=2), vlabels)
                running_false_positives += ((predicate_matrix[vlabels] - voutputs) == -1).sum()

        avg_vloss = running_vloss / (i + 1)
        avg_acc = running_acc / (i + 1)
        avg_false_positives = running_false_positives / (i + 1) / len(validation_loader.dataset)
        print(f"Loss: {avg_vloss}, ACC: {avg_acc}, FP: {avg_false_positives}")

    # save stats to csv
    #with open("LoopStats2.csv", "a") as f:
        #f.write(f"{timestamp},{FT_WEIGHT},{POS_FT_WEIGHT},{best_stats['epoch']},{best_stats['train_loss']},{best_stats['val_loss']},{best_stats['val_acc']},{best_stats['val_fp']}\n")
