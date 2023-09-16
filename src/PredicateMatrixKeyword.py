# ====================================================
# get model
print("Getting model...")
import sys
sys.path.insert(0, "/".join(__file__.split("/")[:-1]) + "/models")
from ResnetAutoPredicates import ResExtr

NUM_FEATURES = 64
NUM_CLASSES = 200

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResExtr(NUM_FEATURES, NUM_CLASSES, resnet_type=18, pretrained=True).to(device)
model.load_state_dict(torch.load("CUBRes18AutoPred.pt"))
model.eval()

predicate_matrix = model.bin_quantize(model.predicate_matrix.view(-1, 1))[0].view(NUM_CLASSES, NUM_FEATURES)

# ====================================================
# for each class, get keywords from description

print("Getting class attributes...")
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from rake_nltk import Rake
rake_nltk_var = Rake()

class_keywords = {}

attributes_root = "results/CUB-Attributes" # each file is {class_name}.txt
import os

for filename in os.listdir(attributes_root):
    class_name = filename[:-4]
    description_keywords = set()
    with open(os.path.join(attributes_root, filename)) as f:
        lines = f.read().split("\n")
        for line in lines:
            if not line.startswith("- "):
                continue
            rake_nltk_var.extract_keywords_from_text(line)
            description_keywords.update(rake_nltk_var.get_ranked_phrases())

    # clean keywords
    cleaned_description_keywords = set()
    for keyword in description_keywords:
        while keyword and not keyword[-1].isalnum():
            keyword = keyword[:-1]
        while keyword and not keyword[0].isalnum():
            keyword = keyword[1:]
        cleaned_description_keywords.add(keyword)

    class_keywords[class_name] = cleaned_description_keywords

# ====================================================
# get classes name

print("Getting class names...")
classes_path = "data/CUB_200_2011/CUB_200_2011/classes.txt"

classes = []

with open(classes_path) as f:
    classes_lines = f.readlines()

for line in classes_lines:
    classes.append(line.split(".")[1].strip().replace("_", " ").lower())

# ====================================================
# for each attribute, get most common keyword

print("Getting attribute keywords...")

predicate_keywords = []
for i in range(NUM_FEATURES):
    feature_column = predicate_matrix[:, i]
    activated_classes = [i for i, x in enumerate(feature_column) if x == 1]
    
    all_keywords = {} # keyword: count
    for activated_class in activated_classes:
        class_name = classes[activated_class]
        for keyword in class_keywords[class_name]:
            if keyword in all_keywords:
                all_keywords[keyword] += 1
            else:
                all_keywords[keyword] = 1

    sorted_keywords = sorted(all_keywords.items(), key=lambda x: x[1], reverse=True)
    # remove single word nouns
    final_keywords = []
    for feature_attribute in sorted_keywords:
        if len(feature_attribute[0].split()) > 1:
            if feature_attribute[0] in ["sized bird", "sized songbird", "breeding season"]:
                continue
            final_keywords.append(feature_attribute)
        elif not feature_attribute[0] in ["tail", "face", "sides", "females", "side", "top", "similar", "underparts", "female", "eyes", "body", "males", "head", "wings", "chest", "bird", "eye", "neck", "male", "breast", "back", "bill", "throat", "flanks", "belly", "feathers"]:
            final_keywords.append(feature_attribute)

    sorted_keywords = final_keywords
    predicate_keywords.append(sorted_keywords[:20])

# ====================================================
target_feature = 2
print(predicate_keywords[target_feature])
