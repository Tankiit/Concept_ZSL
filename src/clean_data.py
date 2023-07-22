import numpy as np

# load data from file
predicates_matrix = np.loadtxt('datasets/Animals_with_Attributes2/predicate-matrix-binary.txt', dtype=int)

features = np.loadtxt('datasets/Animals_with_Attributes2/predicates.txt', dtype=str)[:, 1]
classes = np.loadtxt('datasets/Animals_with_Attributes2/classes.txt', dtype=str)[:, 1]

# create feature bank
feature_bank = {}

for i, predicate in enumerate(predicates_matrix):
    class_ = classes[i]
    feature_bank[class_] = []
    for j, feature in enumerate(predicate):
        if feature == 1:
            feature_bank[class_].append(j)

for class_, feature_list in feature_bank.items():
    feature_bank[class_] = [set(feature_list)]

from Tree import extand_feature_bank

feature_bank = extand_feature_bank(feature_bank)

# save data as json
feature_bank = {class_: [list(feature_set) for feature_set in feature_list] for class_, feature_list in feature_bank.items()}

data = {
    "classes": classes.tolist(),
    "features": features.tolist(),
    "feature_bank": feature_bank
}

import json
with open('datasets/Animals_with_Attributes2/data.json', 'w') as f:
    json.dump(data, f)
