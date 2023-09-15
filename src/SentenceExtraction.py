print("Getting attributes...")

attributes_root = "results/CUB-IMGAttr"

FEATURES = 64

import os

feature_attributes = {}

for filename in os.listdir(attributes_root):
    sentence_extracted = []
    activated_attributes = []
    with open(os.path.join(attributes_root, filename)) as f:
        lines = f.read().split("\n")
        activated_attributes = [i for i, line in enumerate(lines[1:65]) if line == "1.0"]
        for line in lines[67:]:
            if not line:
                continue
            sentence_extracted.append(line)

    for sentence in sentence_extracted:
        if sentence in feature_attributes:
            for activated_attribute in activated_attributes:
                feature_attributes[sentence][activated_attribute] += 1
        else:
            feature_attributes[sentence] = [0] * FEATURES
            for activated_attribute in activated_attributes:
                feature_attributes[sentence][activated_attribute] += 1

# get most common attribute for feature N
target_feature = 2
sorted_feature_attributes = sorted(feature_attributes.items(), key=lambda x: x[1][target_feature], reverse=True)

ATTRIBUTE_COUNT = 10

most_common_attributes = [(x[1][target_feature],x[0]) for x in sorted_feature_attributes[:ATTRIBUTE_COUNT]]

print(most_common_attributes)