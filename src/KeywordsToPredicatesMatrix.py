"""

Create a dictionary of {class: [keywords]}

Take the num_keywords most common words and create a matrix of shape (num_classes, num_keywords)

"""

import os, nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from rake_nltk import Rake
rake_nltk_var = Rake()

def get_keywords_from_file(file_path):
    with open(file_path) as f:
        lines = f.read().split("\n")
        text = ""
        for line in lines[1:]:
            if not line:
                continue
            text += line[2:] + ". " if line[-1] != "." else line[2:]

    rake_nltk_var.extract_keywords_from_text(text)
    keyword_extracted = set(rake_nltk_var.get_ranked_phrases())

    # remove any punctuation at the end of the keyword like "yellow )." or "yellow ."
    cleaned_keyword_extracted = []
    for keyword in keyword_extracted:
        while keyword and not keyword[-1].isalnum():
            keyword = keyword[:-1]

        if keyword and len(keyword) > 1:
            cleaned_keyword_extracted.append(keyword)

    keyword_extracted = cleaned_keyword_extracted

    to_remove = ["sized bird", "members", "neck", "slightly", "long", "legs", "others", "sleeve", "around neck", "adults", "sized songbird", "breeding season", "tail", "face", "sides", "females", "side", "top", "similar", "underparts", "female", "eyes", "body", "males", "head", "wings", "chest", "bird", "eye", "neck", "male", "breast", "back", "bill", "throat", "flanks", "belly", "feathers"]

    # remove single word nouns
    final_feature_attributes = []
    for feature_attribute in keyword_extracted:
        if not feature_attribute in to_remove:
            final_feature_attributes.append(feature_attribute)

    return final_feature_attributes

def get_keywords_from_folder(folder_path):
    # Make dictionary of {file_name: [keywords]}
    file_keywords = {}
    for file_name in os.listdir(folder_path):
        if not file_name.endswith(".txt"):
            continue
        file_path = os.path.join(folder_path, file_name)
        file_keywords[file_name[:-4]] = get_keywords_from_file(file_path)

    return file_keywords

import torch
def build_predicate_matrix(sorted_keyword_occurence, keywords_dict, N):
    predicate_matrix = torch.zeros((len(keywords_dict), N))

    for i, (keyword, _) in enumerate(sorted_keyword_occurence[:N]):
        for j, (_, keywords) in enumerate(keywords_dict.items()):
            if keyword in keywords:
                predicate_matrix[j][i] = 1

    # Make sure no row is subset of another row

    for i in range(predicate_matrix.shape[0]):
        for j in range(predicate_matrix.shape[0]):
            if i == j:
                continue
            
            if torch.all(predicate_matrix[i] <= predicate_matrix[j]):
                return None
            
    return predicate_matrix

def keywords_to_predicate_mat(keywords_dict, N):
    keyword_occurence = {}
    for keywords in keywords_dict.values():
        for keyword in keywords:
            if keyword in keyword_occurence:
                keyword_occurence[keyword] += 1
            else:
                keyword_occurence[keyword] = 1

    sorted_keyword_occurence = sorted(keyword_occurence.items(), key=lambda x: x[1], reverse=True)

    predicate_matrix = build_predicate_matrix(sorted_keyword_occurence, keywords_dict, N)
    while predicate_matrix is None:
        N += 1
        predicate_matrix = build_predicate_matrix(sorted_keyword_occurence, keywords_dict, N)

    return predicate_matrix

if __name__ == "__main__":
    folder = "results/CUB-Attributes"
    file_keywords = get_keywords_from_folder(folder)

    preds = keywords_to_predicate_mat(file_keywords, 32)
    print(preds.shape)