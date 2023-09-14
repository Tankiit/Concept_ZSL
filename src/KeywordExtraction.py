import nltk
nltk.download('stopwords')
nltk.download('punkt')

from rake_nltk import Rake
rake_nltk_var = Rake()

print("Getting attributes...")

attributes_root = "results/CUB-IMGAttr"

FEATURES = 64

import os

feature_attributes = {}

for filename in os.listdir(attributes_root):
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
        if keyword in activated_attributes:
            for activated_attribute in activated_attributes:
                feature_attributes[keyword][activated_attribute] += 1

        else:
            feature_attributes[keyword] = [0] * FEATURES
            for activated_attribute in activated_attributes:
                feature_attributes[keyword][activated_attribute] += 1

# get most common attribute for feature N
target_feature = 2
sorted_feature_attributes = sorted(feature_attributes.items(), key=lambda x: x[1][target_feature], reverse=True)

ATTRIBUTE_COUNT = 30

most_common_attributes = [x[0] for x in sorted_feature_attributes[:ATTRIBUTE_COUNT]]

print(most_common_attributes)
