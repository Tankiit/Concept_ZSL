import torch.nn as nn
import torch
from ResnetExtraction import ResExtr
from torchmetrics import Accuracy
from AwA2Loader import get_train_test_zsl

def loss_fn(out, y_predicates, y_classes, ft_weight=.1):
    out = out.view(-1, 1, NUM_FEATURES)
    ANDed = out * y_predicates
    diff = ANDed - out

    entr_loss = nn.CrossEntropyLoss()
    loss_cl = entr_loss(diff.sum(dim=2), y_classes)

    #batch_size = out.shape[0]

    #labels = torch.randint(0, num_classes-1, (batch_size, ), device="cuda")
    #classes = torch.zeros(batch_size, num_classes, device="cuda")
    #classes[torch.arange(batch_size), labels] = 1
    #classes = classes.view(batch_size, num_classes, 1).expand(batch_size, num_classes, num_features)

    #extra_features = out - y_predicates + (out - y_predicates).pow(2)

    #loss_ft = torch.masked_select(extra_features, (1-classes).bool()).view(-1, num_features).sum() / batch_size

    return loss_cl #+ loss_ft * ft_weight * loss_cl.item()/loss_ft.item()

def train_one_epoch():
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data['features'].to(device), data['labels'].to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs, commit_loss = model(inputs)

        # Compute the loss and its gradients
        print(labels.min(), labels.max())
        loss = loss_fn(outputs, train_predicate_matrix, labels) + commit_loss
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss

from datetime import datetime
if __name__ == "__main__":
    feature_file = "datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt"
    label_file = "datasets/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt"
    predicate_matrix_file = "datasets/Animals_with_Attributes2/predicate-matrix-binary.txt"
    train_classes_file = "datasets/Animals_with_Attributes2/Features/ResNet101/trainclasses.txt"
    classes_labels_file = "datasets/Animals_with_Attributes2/classes.txt"
    train_set, val_set, train_predicate_matrix, test_predicate_matrix = get_train_test_zsl(feature_file, label_file, predicate_matrix_file, classes_labels_file, train_classes_file)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_predicate_matrix = train_predicate_matrix.to(device)
    test_predicate_matrix = test_predicate_matrix.to(device)

    training_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=True, num_workers=4)
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)

    NUM_FEATURES = 85

    model = ResExtr(2048, NUM_FEATURES, 1).to(device)

    accuracy = Accuracy(task="multiclass", num_classes=10, top_k=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch()

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        acc = 0.0

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata['features'].to(device), vdata['labels'].to(device)
                voutputs, vcommit_loss = model(vinputs)
                vloss = loss_fn(voutputs, test_predicate_matrix, vlabels) + vcommit_loss
                running_vloss += vloss
                ANDed = voutputs * test_predicate_matrix
                diff = ANDed - voutputs
                acc = accuracy(diff.sum(dim=2), vlabels)
                print(acc)

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}, ACC {}'.format(avg_loss, avg_vloss, acc))

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1