import torch
from torch.utils.data import Dataset
from numpy import loadtxt, float32
class AwA2ResNetDataset(Dataset):
    def __init__(self, feature_file, label_file, predicate_matrix_file, file_paths_file):
        self.features = torch.from_numpy(loadtxt(feature_file, dtype=float32))
        self.labels = torch.from_numpy(loadtxt(label_file, dtype=int))
        self.predicate_matrix = torch.from_numpy(loadtxt(predicate_matrix_file, dtype=int))
        self.file_paths = open(file_paths_file, "r").read().split("\n")[:-1]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'labels': self.labels[idx] - 1, 'predicate_matrix': self.predicate_matrix}

class AwA2ResNetDatasetZSL(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {'features': self.features[idx], 'labels': self.labels[idx] - 1}

def get_train_test_zsl(feature_file, label_file, predicate_matrix_file, classes_labels_file, train_classes_file):
    train_classes = open(train_classes_file, "r").read().replace("+", "_").split("\n")[:-1]

    classes_labels = {}
    for line in open(classes_labels_file, "r").read().split("\n")[:-1]:
        label, class_ = line.split("\t")
        classes_labels[class_] = int(label)

    train_classes_labels = []
    for train_class in train_classes:
        train_classes_labels.append(classes_labels[train_class])
        
    test_classes_labels = []
    for i in range(1, 51):
        if i not in train_classes_labels:
            test_classes_labels.append(i)

    features = torch.from_numpy(loadtxt(feature_file, dtype=float32))
    labels = torch.from_numpy(loadtxt(label_file, dtype=int))
    predicate_matrix = torch.from_numpy(loadtxt(predicate_matrix_file, dtype=int))

    # get all indices of labels that are in train_classes_labels
    train_indices = []
    for i, label in enumerate(labels):
        if label in train_classes_labels:
            train_indices.append(i)

    train_indices = torch.tensor(train_indices)
    train_features = features[train_indices]
    train_labels = labels[train_indices]
    train_predicate_matrix = predicate_matrix[torch.tensor(train_classes_labels)-1]

    test_indices = []
    for i, label in enumerate(labels):
        if label in test_classes_labels:
            test_indices.append(i)

    test_indices = torch.tensor(test_indices)
    test_features = features[test_indices]
    test_labels = labels[test_indices]
    test_predicate_matrix = predicate_matrix[torch.tensor(test_classes_labels)-1]

    return AwA2ResNetDatasetZSL(train_features, train_labels), AwA2ResNetDatasetZSL(test_features, test_labels), train_predicate_matrix, test_predicate_matrix