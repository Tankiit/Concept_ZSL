import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from rake_nltk import Rake
rake_nltk_var = Rake()

print("Getting attributes...")

attributes_root = "results/CUB-IMGAttr"

FEATURES = 64

import os

feature_attributes = {}

for filename in os.listdir(attributes_root):
    cleaned_keyword_extracted = []
    activated_attributes = []
    with open(os.path.join(attributes_root, filename)) as f:
        lines = f.read().split("\n")
        activated_attributes = [i for i, line in enumerate(lines[1:65]) if line == "1.0"]
        text = ""
        for line in lines[67:]:
            if not line:
                continue
            text += line + ". " if line[-1] != "." else line

    rake_nltk_var.extract_keywords_from_text(text)
    keyword_extracted = set(rake_nltk_var.get_ranked_phrases())

    # remove any punctuation at the end of the keyword like "yellow )." or "yellow ."
    cleaned_keyword_extracted = []
    for keyword in keyword_extracted:
        while keyword and not keyword[-1].isalnum():
            keyword = keyword[:-1]
        cleaned_keyword_extracted.append(keyword)

    keyword_extracted = cleaned_keyword_extracted
    
    for keyword in keyword_extracted:
        if keyword in feature_attributes:
            for activated_attribute in activated_attributes:
                feature_attributes[keyword][activated_attribute] += 1
        else:
            feature_attributes[keyword] = [0] * FEATURES
            for activated_attribute in activated_attributes:
                feature_attributes[keyword][activated_attribute] += 1

# get most common attribute for feature N
target_feature = 55
sorted_feature_attributes = sorted(feature_attributes.items(), key=lambda x: x[1][target_feature], reverse=True)

to_remove = ["sized bird", "sleeve", "around neck", "adults", "sized songbird", "breeding season", "tail", "face", "sides", "females", "side", "top", "similar", "underparts", "female", "eyes", "body", "males", "head", "wings", "chest", "bird", "eye", "neck", "male", "breast", "back", "bill", "throat", "flanks", "belly", "feathers"]

# remove single word nouns
final_feature_attributes = []
for feature_attribute in sorted_feature_attributes:
    if not feature_attribute[0] in to_remove:
        final_feature_attributes.append(feature_attribute)

ATTRIBUTE_COUNT = 50

most_common_attributes = [x[0] for x in final_feature_attributes[:ATTRIBUTE_COUNT]]

# only keep attributes that are above the average by more than the standard deviation
def mean(lst):
    return sum(lst) / len(lst)

def stdev(lst):
    avg = mean(lst)
    return avg, (sum([(x - avg) ** 2 for x in lst]) / len(lst)) ** 0.5

final_attributes = []
for attribute in most_common_attributes:
    counts = feature_attributes[attribute]
    avg, std = stdev(counts)
    if counts[target_feature] > avg + std*0.5:
        final_attributes.append(attribute)

final_attributes = [(feature_attributes[x][target_feature], mean(feature_attributes[x]),x) for x in final_attributes]

print(f"Most common attributes for feature {target_feature}:")
print(final_attributes)

print("=====================================")

CMAO_attributes = sorted(feature_attributes.items(), key=lambda x: x[1][target_feature] / mean(x[1]), reverse=True)

final_feature_attributes = []
for feature_attribute in CMAO_attributes:
    if not feature_attribute[0] in to_remove:
        final_feature_attributes.append(feature_attribute)

threshold = 30
CMAO_attributes = [(feature_attributes[x[0]][target_feature], mean(feature_attributes[x[0]]),x[0]) for x in final_feature_attributes if feature_attributes[x[0]][target_feature] > threshold]

print(CMAO_attributes[:ATTRIBUTE_COUNT])

print("=====================================")

print(most_common_attributes)
