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
# get image attributes

image_root = "data/CUB_200_2011"
image = "075.Green_Jay/Green_Jay_0111_65869.jpg"

from torchvision.transforms.functional import normalize, resize
from torchvision.io.image import read_image
import os

img = read_image(os.path.join(image_root, 'CUB_200_2011/images', image))
input_tensor = normalize(resize(img, (224, 224), antialias=True) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to(device)
voutputs, _, _ = model(input_tensor.unsqueeze(0))

active_attributes = [i for i, voutput in enumerate(voutputs[0]) if voutput > 0.5]
label = int(image.split(".")[0]) - 1
class_predicates = predicate_matrix[label]
print(f"Predicates match: {(voutputs == class_predicates).sum().item()}")

# ====================================================
# get image attributes
print("Getting attributes...")
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

from rake_nltk import Rake
rake_nltk_var = Rake()

description_keywords = set()

attributes_root = "results/CUB-IMGAttr"

FEATURES = 64

import os

feature_attributes = {}

for filename in os.listdir(attributes_root):
    clean_keyword_extracted = []
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

common_attributes = []
for target_feature in active_attributes:
    sorted_feature_attributes = sorted(feature_attributes.items(), key=lambda x: x[1][target_feature], reverse=True)
    
    # remove single word nouns
    final_feature_attributes = []
    for feature_attribute in sorted_feature_attributes:
        if len(feature_attribute[0].split()) > 1:
            if feature_attribute[0] in ["sized bird", "sized songbird", "breeding season"]:
                continue
            final_feature_attributes.append(feature_attribute)
        elif not feature_attribute[0] in ["tail", "face", "sides", "females", "side", "top", "similar", "underparts", "female", "eyes", "body", "males", "head", "wings", "chest", "bird", "eye", "neck", "male", "breast", "back", "bill", "throat", "flanks", "belly", "feathers"]:
            final_feature_attributes.append(feature_attribute)

    ATTRIBUTE_COUNT = 20

    most_common_attributes = [[x[1][target_feature],x[0]] for x in final_feature_attributes[:ATTRIBUTE_COUNT]]

    for st in most_common_attributes:
        for st2 in common_attributes:
            if st[1] == st2[1]:
                st2[0] += st[0]
                break
        else:
            common_attributes.append(st)

common_attributes.sort(key=lambda x: x[0], reverse=True)
print(common_attributes)

# ====================================================
# clean with CLIP
print("Cleaning with CLIP...")
from tqdm import tqdm
from PIL import Image

import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)

captions = [x[1] for x in common_attributes]

pil_image = Image.open(os.path.join(image_root, 'CUB_200_2011/images', image))
image = preprocess(pil_image).unsqueeze(0).to(device)

with torch.no_grad():
    text = clip.tokenize(captions).to(device)
    logits_per_image, logits_per_text = model(image, text)

# get topk captions
TOPK = 15

probs = logits_per_image.softmax(dim=-1).cpu().numpy()
topk = probs[0].argsort()[-TOPK:][::-1]
topk_captions = [captions[x] for x in topk]

print(topk_captions)
# ====================================================
# display image

pil_image.show()