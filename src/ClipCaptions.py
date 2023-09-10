# import model

print("Loading model...")

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

# ==================================================================================================
# get image paths

print("Getting images...")

import pandas as pd
import os

image_root = "data/CUB_200_2011"

images = pd.read_csv(os.path.join(image_root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
image_class_labels = pd.read_csv(os.path.join(image_root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
data = images.merge(image_class_labels, on='img_id')
train_test_split = pd.read_csv(os.path.join(image_root, 'CUB_200_2011', 'train_test_split.txt'),
                               sep=' ', names=['img_id', 'is_training_img'])

data = data.merge(train_test_split, on='img_id')

data = data[data.is_training_img == 0]
images = data["filepath"].tolist()

# ==================================================================================================
# get COCO captions

print("Getting captions...")

import json
caption_root = "data/annotations_trainval2014/annotations"

with open(os.path.join(caption_root, "captions_train2014.json")) as f:
    coco_train = json.load(f)

with open(os.path.join(caption_root, "captions_val2014.json")) as f:
    coco_val = json.load(f)

coco = coco_train["annotations"] + coco_val["annotations"]
captions = [x["caption"] for x in coco]

# ==================================================================================================
# compare captions to images using CLIP

from tqdm import tqdm
from PIL import Image

TOPK = 100

import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

print("Encoding captions...")

MAX_CAPTIONS = 3000
CAPTION_BUCKETS = len(captions) // MAX_CAPTIONS + 1

print("Comparing images to captions...")
for i, image_path in tqdm(enumerate(images)):
    pil_image = Image.open(os.path.join(image_root, 'CUB_200_2011/images', image_path))
    image = preprocess(pil_image).unsqueeze(0).to(device)

    all_probs = []
    with torch.no_grad():
        for j in range(CAPTION_BUCKETS):
            text = clip.tokenize(captions[j*MAX_CAPTIONS:(j+1)*MAX_CAPTIONS]).to(device)
            logits_per_image, logits_per_text = model(image, text)
            all_probs.append(logits_per_image)

    # get topk captions
    probs = torch.cat(all_probs, dim=1).softmax(dim=-1).cpu().numpy()
    topk = probs[0].argsort()[-TOPK:][::-1]
    topk_captions = [captions[x] for x in topk]

    print(topk_captions)

    # display image
    pil_image.show()

    exit()