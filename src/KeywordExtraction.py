import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

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
    keyword_extracted = rake_nltk_var.get_ranked_phrases()

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
target_feature = 1
sorted_feature_attributes = sorted(feature_attributes.items(), key=lambda x: x[1][target_feature], reverse=True)

# remove single word nouns
final_feature_attributes = []
for feature_attribute in sorted_feature_attributes:
    if len(feature_attribute[0].split()) > 1:
        final_feature_attributes.append(feature_attribute)
    elif not feature_attribute[0] in ["tail", "head", "wings", "chest", "bird", "eye", "neck", "male", "breast", "back", "bill", "throat", "flanks", "belly"]:
        final_feature_attributes.append(feature_attribute)

sorted_feature_attributes = final_feature_attributes

ATTRIBUTE_COUNT = 50

most_common_attributes = [x[0] for x in sorted_feature_attributes[:ATTRIBUTE_COUNT]]

print(most_common_attributes)
